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
    let (inner, _) = unwrap_vertex_style(v);
    let key = expr_to_string(inner);
    let idx = graph.add_node(i);
    index_map.insert(key, idx);
  }

  for e in raw_edges {
    let (edge_expr, _label, _color) = unwrap_edge_wrappers(e);
    match edge_expr {
      Expr::FunctionCall {
        name: ename,
        args: eargs,
      } if eargs.len() == 2 => {
        let directed = ename == "DirectedEdge";
        let src_key = expr_to_string(unwrap_vertex_style(&eargs[0]).0);
        let dst_key = expr_to_string(unwrap_vertex_style(&eargs[1]).0);
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
        let src_key = expr_to_string(unwrap_vertex_style(pattern).0);
        let dst_key = expr_to_string(unwrap_vertex_style(replacement).0);
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
    let (inner, _) = unwrap_vertex_style(v);
    let key = expr_to_string(inner);
    let idx = graph.add_node(i);
    index_map.insert(key, idx);
  }

  for edge in edges {
    let (edge_expr, _, _) = unwrap_edge_wrappers(edge);
    if let Expr::FunctionCall { args: eargs, .. } = edge_expr
      && eargs.len() == 2
    {
      let from_str = expr_to_string(unwrap_vertex_style(&eargs[0]).0);
      let to_str = expr_to_string(unwrap_vertex_style(&eargs[1]).0);
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
    let (inner, _) = unwrap_vertex_style(v);
    let key = expr_to_string(inner);
    let idx = graph.add_node(i);
    index_map.insert(key, idx);
  }

  for edge in edges {
    let (edge_expr, _, _) = unwrap_edge_wrappers(edge);
    if let Expr::FunctionCall {
      name: ename,
      args: eargs,
    } = edge_expr
      && eargs.len() == 2
    {
      let from_str = expr_to_string(unwrap_vertex_style(&eargs[0]).0);
      let to_str = expr_to_string(unwrap_vertex_style(&eargs[1]).0);
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
  color: Option<Color>,
}

/// Build a petgraph DiGraph for rendering (preserves edge labels and direction info).
fn build_render_graph(
  vertices: &[Expr],
  raw_edges: &[Expr],
) -> (DiGraph<usize, RenderEdgeData>, HashMap<String, NodeIndex>) {
  let mut graph = DiGraph::new();
  let mut index_map: HashMap<String, NodeIndex> = HashMap::new();

  for (i, v) in vertices.iter().enumerate() {
    let (inner, _) = unwrap_vertex_style(v);
    let key = expr_to_output(inner);
    let idx = graph.add_node(i);
    index_map.insert(key, idx);
  }

  for e in raw_edges {
    let (edge_expr, label, color) = unwrap_edge_wrappers(e);
    if let Expr::FunctionCall {
      name: ename,
      args: eargs,
    } = edge_expr
      && eargs.len() == 2
    {
      let directed = ename == "DirectedEdge";
      let src_key = expr_to_output(unwrap_vertex_style(&eargs[0]).0);
      let dst_key = expr_to_output(unwrap_vertex_style(&eargs[1]).0);
      if let (Some(&si), Some(&di)) =
        (index_map.get(&src_key), index_map.get(&dst_key))
      {
        graph.add_edge(
          si,
          di,
          RenderEdgeData {
            directed,
            label: label.clone(),
            color,
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
      args: args.to_vec().into(),
    });
  }

  let vertices = match &args[0] {
    Expr::List(v) => v.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let raw_edges = match &args[1] {
    Expr::List(e) => e.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let n = vertices.len();
  if n == 0 {
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: args.to_vec().into(),
    });
  }

  // Parse options from remaining args
  let options = &args[2..];
  let mut vertex_style: Option<Vec<Expr>> = None;
  let mut edge_style: Option<Color> = None;
  let mut vertex_labels = false;
  let mut vertex_shape: Option<String> = None;
  let mut vertex_size_scale: f64 = 1.0;
  let mut plot_label: Option<Expr> = None;

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
          // Named sizes use progressively larger gaps so a Table over
          // {Tiny, Small, Medium, Large} produces a visibly increasing
          // sequence of vertex sizes. Tiny is kept at the historical 0.5
          // so existing "barely-a-dot" renderings are preserved.
          vertex_size_scale = match replacement.as_ref() {
            Expr::Identifier(s) => match s.as_str() {
              "Tiny" => 0.6,
              "Small" => 1.2,
              "Medium" => 2.2,
              "Large" => 3.6,
              _ => 1.0,
            },
            _ => crate::functions::graphics::expr_to_f64(replacement)
              .unwrap_or(1.0),
          };
        }
        "PlotLabel" => {
          // Ignore PlotLabel -> None / PlotLabel -> Null so that defaults
          // don't accidentally render an empty label.
          let is_none = matches!(
            replacement.as_ref(),
            Expr::Identifier(s) if s == "None" || s == "Null"
          );
          if !is_none {
            plot_label = Some((**replacement).clone());
          }
        }
        _ => {}
      }
    }
  }

  // Build petgraph for rendering
  let (graph, _index_map) = build_render_graph(&vertices, &raw_edges);

  // Compute vertex positions. For a single weakly-connected component we
  // keep the simple circular embedding; for multi-component graphs each
  // component is laid out independently (force-directed when large enough)
  // and the components are packed into a grid so clusters are visible.
  let positions: Vec<(f64, f64)> = compute_layout(&graph);

  // Compute base radius for vertices. Kept deliberately small so labels
  // and edges remain legible; a border is drawn around each vertex so the
  // shape stays visible even at this size.
  let base_radius = if n <= 2 {
    0.06
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
    (min_dist * 0.08).min(0.06).max(0.018)
  };
  let vertex_radius = base_radius * vertex_size_scale;

  // Build Graphics primitives
  let mut primitives: Vec<Expr> = Vec::new();

  // --- Draw edges ---
  let default_edge_color = Color::new(0.6, 0.6, 0.6);
  let default_applied_edge_color =
    *edge_style.as_ref().unwrap_or(&default_edge_color);
  primitives.push(default_applied_edge_color.to_expr());
  primitives.push(Expr::FunctionCall {
    name: "AbsoluteThickness".to_string(),
    args: vec![Expr::Real(1.5)].into(),
  });

  // Group parallel (including antiparallel) non-loop edges by unordered
  // vertex pair so that multi-edges can be rendered as separate curves
  // instead of overlapping on the same straight line. The position of an
  // edge within its group determines its perpendicular offset.
  let mut parallel_groups: std::collections::HashMap<
    (usize, usize),
    Vec<petgraph::graph::EdgeIndex>,
  > = std::collections::HashMap::new();
  for edge_ref in graph.edge_references() {
    let si = edge_ref.source().index();
    let di = edge_ref.target().index();
    if si == di {
      continue;
    }
    let key = (si.min(di), si.max(di));
    parallel_groups.entry(key).or_default().push(edge_ref.id());
  }

  for edge_ref in graph.edge_references() {
    let si = edge_ref.source().index();
    let di = edge_ref.target().index();
    let edge_data = edge_ref.weight();
    let (x1, y1) = positions[si];
    let (x2, y2) = positions[di];

    // Apply per-edge color when present, otherwise restore the default
    // edge color so a previous styled edge does not bleed into this one.
    if let Some(c) = &edge_data.color {
      primitives.push(c.to_expr());
    } else {
      primitives.push(default_applied_edge_color.to_expr());
    }

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
          pts.push(Expr::List(vec![Expr::Real(px), Expr::Real(py)].into()));
        } else {
          let px = cx + loop_r * angle.cos();
          let py = cy + loop_r * angle.sin();
          pts.push(Expr::List(vec![Expr::Real(px), Expr::Real(py)].into()));
        }
      }
      if edge_data.directed {
        primitives.push(Expr::FunctionCall {
          name: "Arrow".to_string(),
          args: vec![Expr::List(pts.into())].into(),
        });
      } else {
        primitives.push(Expr::FunctionCall {
          name: "Line".to_string(),
          args: vec![Expr::List(pts.into())].into(),
        });
      }
    } else {
      // Determine this edge's position within its parallel group.
      // A group of size 1 → straight edge (current behavior).
      // A group of size ≥ 2 → quadratic Bézier curve with a
      // perpendicular offset that is unique to this edge's position.
      let key = (si.min(di), si.max(di));
      let group = &parallel_groups[&key];
      let k_in_group = group
        .iter()
        .position(|&id| id == edge_ref.id())
        .unwrap_or(0);
      let total = group.len();

      // Canonical perpendicular direction, computed from the low→high
      // ordering so that antiparallel edges pick the *same* perpendicular
      // axis and therefore curve on opposite physical sides automatically
      // when rendered with the actual source→target direction.
      let (lo, hi) = key;
      let (lx, ly) = positions[lo];
      let (hx, hy) = positions[hi];
      let cdx = hx - lx;
      let cdy = hy - ly;
      let clen = (cdx * cdx + cdy * cdy).sqrt().max(1e-9);
      let perp_x = -cdy / clen;
      let perp_y = cdx / clen;

      // Offset index centered around 0. With total=2 we get [-0.5, +0.5];
      // with total=3 we get [-1, 0, +1]; etc.
      let offset_idx = k_in_group as f64 - (total as f64 - 1.0) / 2.0;
      let spacing = vertex_radius * 1.4;
      let offset_mag = offset_idx * spacing;

      if total == 1 || offset_mag.abs() < 1e-9 {
        // Straight edge — unchanged behavior.
        if edge_data.directed {
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
            args: vec![Expr::List(
              vec![
                Expr::List(vec![Expr::Real(sx), Expr::Real(sy)].into()),
                Expr::List(vec![Expr::Real(ex), Expr::Real(ey)].into()),
              ]
              .into(),
            )]
            .into(),
          });
        } else {
          primitives.push(Expr::FunctionCall {
            name: "Line".to_string(),
            args: vec![Expr::List(
              vec![
                Expr::List(vec![Expr::Real(x1), Expr::Real(y1)].into()),
                Expr::List(vec![Expr::Real(x2), Expr::Real(y2)].into()),
              ]
              .into(),
            )]
            .into(),
          });
        }
      } else {
        // Curved edge: quadratic Bézier through a perpendicular-offset
        // control point. The same perp axis is used regardless of which
        // direction the edge travels, so a pair (a→b, b→a) with indices
        // 0 and 1 in the group get offsets -0.5 and +0.5 respectively
        // and render as two clearly distinct curves on opposite sides.
        let ctrl_x = (x1 + x2) / 2.0 + perp_x * offset_mag;
        let ctrl_y = (y1 + y2) / 2.0 + perp_y * offset_mag;

        // Shorten each endpoint toward the control point by vertex_radius
        // so the curve starts/ends on the vertex boundary with its tangent
        // pointing into the curve.
        let shorten = |vx: f64, vy: f64| -> (f64, f64) {
          let dx = ctrl_x - vx;
          let dy = ctrl_y - vy;
          let d = (dx * dx + dy * dy).sqrt().max(1e-9);
          (vx + dx / d * vertex_radius, vy + dy / d * vertex_radius)
        };
        let (sx, sy) = shorten(x1, y1);
        let (ex, ey) = shorten(x2, y2);

        let segments = 18;
        let mut pts = Vec::with_capacity(segments + 1);
        for k in 0..=segments {
          let t = k as f64 / segments as f64;
          let omt = 1.0 - t;
          let bx = omt * omt * sx + 2.0 * omt * t * ctrl_x + t * t * ex;
          let by = omt * omt * sy + 2.0 * omt * t * ctrl_y + t * t * ey;
          pts.push(Expr::List(vec![Expr::Real(bx), Expr::Real(by)].into()));
        }

        if edge_data.directed {
          primitives.push(Expr::FunctionCall {
            name: "Arrow".to_string(),
            args: vec![Expr::List(pts.into())].into(),
          });
        } else {
          primitives.push(Expr::FunctionCall {
            name: "Line".to_string(),
            args: vec![Expr::List(pts.into())].into(),
          });
        }
      }
    }

    // Edge label
    if let Some(lbl) = &edge_data.label {
      let mx = if si == di { x1 } else { (x1 + x2) / 2.0 };
      let my = if si == di {
        y1 + vertex_radius * 2.0 + vertex_radius * 1.8 * 1.5
      } else {
        (y1 + y2) / 2.0
      };
      // Force labels to black so they don't inherit the edge color.
      primitives.push(Expr::FunctionCall {
        name: "RGBColor".to_string(),
        args: vec![Expr::Real(0.0), Expr::Real(0.0), Expr::Real(0.0)].into(),
      });
      primitives.push(Expr::FunctionCall {
        name: "Text".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Style".to_string(),
            args: vec![Expr::String(lbl.clone()), Expr::Integer(10)].into(),
          },
          Expr::List(vec![Expr::Real(mx), Expr::Real(my)].into()),
        ]
        .into(),
      });
    }
  }

  // --- Draw vertices ---
  let (v_fill, v_edge_form) = parse_vertex_style(&vertex_style);

  if let Some(ef) = &v_edge_form {
    primitives.push(ef.clone());
  } else {
    // Default vertex border: a darker outline so small vertices remain
    // clearly visible and distinguishable from the background.
    primitives.push(Expr::FunctionCall {
      name: "EdgeForm".to_string(),
      args: vec![Expr::List(
        vec![
          Expr::FunctionCall {
            name: "RGBColor".to_string(),
            args: vec![Expr::Real(0.15), Expr::Real(0.27), Expr::Real(0.43)]
              .into(),
          },
          Expr::FunctionCall {
            name: "AbsoluteThickness".to_string(),
            args: vec![Expr::Real(1.0)].into(),
          },
        ]
        .into(),
      )]
      .into(),
    });
  }

  let default_fill =
    Color::new(0.2588235294117647, 0.4549019607843137, 0.7176470588235294); // Wolfram default blue

  // Per-vertex color overrides from `Style[v, color]` wrapping.
  let vertex_colors: Vec<Option<Color>> =
    vertices.iter().map(|v| unwrap_vertex_style(v).1).collect();

  for node_idx in graph.node_indices() {
    let i = node_idx.index();
    let (x, y) = positions[i];

    // Emit this vertex's fill color: per-vertex Style[] override wins over
    // the VertexStyle option, which itself wins over the Wolfram default.
    let fill_for_v: &Color = vertex_colors[i]
      .as_ref()
      .or(v_fill.as_ref())
      .unwrap_or(&default_fill);
    primitives.push(fill_for_v.to_expr());

    match vertex_shape.as_deref() {
      Some("Diamond") => {
        let r = vertex_radius * 1.3;
        primitives.push(Expr::FunctionCall {
          name: "Polygon".to_string(),
          args: vec![Expr::List(
            vec![
              Expr::List(vec![Expr::Real(x), Expr::Real(y + r)].into()),
              Expr::List(vec![Expr::Real(x + r), Expr::Real(y)].into()),
              Expr::List(vec![Expr::Real(x), Expr::Real(y - r)].into()),
              Expr::List(vec![Expr::Real(x - r), Expr::Real(y)].into()),
            ]
            .into(),
          )]
          .into(),
        });
      }
      Some("Square") => {
        let r = vertex_radius * 0.9;
        primitives.push(Expr::FunctionCall {
          name: "Rectangle".to_string(),
          args: vec![
            Expr::List(vec![Expr::Real(x - r), Expr::Real(y - r)].into()),
            Expr::List(vec![Expr::Real(x + r), Expr::Real(y + r)].into()),
          ]
          .into(),
        });
      }
      _ => {
        primitives.push(Expr::FunctionCall {
          name: "Disk".to_string(),
          args: vec![
            Expr::List(vec![Expr::Real(x), Expr::Real(y)].into()),
            Expr::Real(vertex_radius),
          ]
          .into(),
        });
      }
    }

    if vertex_labels {
      // Strip any Style wrapper so the label shows the underlying vertex
      // name (e.g. `3`, not `Style[3, Red]`).
      let label_text = expr_to_output(unwrap_vertex_style(&vertices[i]).0);
      primitives.push(Expr::FunctionCall {
        name: "RGBColor".to_string(),
        args: vec![Expr::Real(0.0), Expr::Real(0.0), Expr::Real(0.0)].into(),
      });
      primitives.push(Expr::FunctionCall {
        name: "Text".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Style".to_string(),
            args: vec![Expr::String(label_text), Expr::Integer(10)].into(),
          },
          Expr::List(
            vec![Expr::Real(x), Expr::Real(y + vertex_radius + 0.08)].into(),
          ),
        ]
        .into(),
      });
    }
  }

  // --- PlotLabel: centered title above the graph ---
  if let Some(label_expr) = plot_label {
    let (x_min, x_max, y_max) = positions.iter().fold(
      (f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
      |(xmin, xmax, ymax), &(x, y)| (xmin.min(x), xmax.max(x), ymax.max(y)),
    );
    let cx = if positions.is_empty() {
      0.0
    } else {
      (x_min + x_max) / 2.0
    };
    let label_y = y_max + vertex_radius + 0.2;

    // If the user already provided a Style[...] wrapper, keep it so their
    // directives (color, size, Bold/Italic) win. Otherwise wrap the label
    // so it renders at a larger, bold font by default.
    let styled = match &label_expr {
      Expr::FunctionCall { name, args }
        if name == "Style" && !args.is_empty() =>
      {
        label_expr.clone()
      }
      _ => Expr::FunctionCall {
        name: "Style".to_string(),
        args: vec![
          Expr::String(expr_to_output(&label_expr)),
          Expr::Integer(16),
          Expr::Identifier("Bold".to_string()),
        ]
        .into(),
      },
    };

    // Force black text regardless of any preceding directive (edges may
    // have set a different color last).
    primitives.push(Expr::FunctionCall {
      name: "RGBColor".to_string(),
      args: vec![Expr::Real(0.0), Expr::Real(0.0), Expr::Real(0.0)].into(),
    });
    primitives.push(Expr::FunctionCall {
      name: "Text".to_string(),
      args: vec![
        styled,
        Expr::List(vec![Expr::Real(cx), Expr::Real(label_y)].into()),
      ]
      .into(),
    });
  }

  let content = Expr::List(primitives.into());
  let image_size_opt = Expr::Rule {
    pattern: Box::new(Expr::Identifier("ImageSize".to_string())),
    replacement: Box::new(Expr::Integer(360)),
  };

  graphics_ast(&[content, image_size_opt])
}

/// Compute vertex positions for the rendered graph.
///
/// Single-component graphs get the existing circular embedding so the
/// output stays stable for the small graphs used in unit tests. Graphs
/// with multiple weakly-connected components are laid out component by
/// component (force-directed once a component is big enough) and the
/// components are packed into a grid, which makes clusters visible — e.g.
/// `Graph[Table[i -> Mod[i^2, 74], {i, 100}]]` renders as 8 clusters.
fn compute_layout(graph: &DiGraph<usize, RenderEdgeData>) -> Vec<(f64, f64)> {
  let n = graph.node_count();
  if n == 0 {
    return vec![];
  }
  if n == 1 {
    return vec![(0.0, 0.0)];
  }
  if n == 2 {
    return vec![(-0.5, 0.0), (0.5, 0.0)];
  }

  let components = weakly_connected_components(graph);

  if components.len() <= 1 {
    return circular_layout(n);
  }

  // Lay out each component independently.
  let mut component_positions: Vec<Vec<(f64, f64)>> =
    Vec::with_capacity(components.len());
  for comp in &components {
    component_positions.push(layout_component(graph, comp));
  }

  pack_components(&components, &component_positions, n)
}

/// Circular layout in [-1, 1]^2 for a connected graph of `n` vertices.
fn circular_layout(n: usize) -> Vec<(f64, f64)> {
  (0..n)
    .map(|k| {
      let angle = PI / 2.0 + (k as f64) * 2.0 * PI / (n as f64);
      (snap_coord(angle.cos()), snap_coord(angle.sin()))
    })
    .collect()
}

/// Weakly connected components of `graph` (directed edges treated as
/// undirected). Components are returned sorted by size in descending order
/// so the largest cluster ends up in a predictable slot of the grid.
fn weakly_connected_components(
  graph: &DiGraph<usize, RenderEdgeData>,
) -> Vec<Vec<usize>> {
  let n = graph.node_count();
  let mut parent: Vec<usize> = (0..n).collect();

  fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    x
  }

  for edge in graph.edge_references() {
    let a = find(&mut parent, edge.source().index());
    let b = find(&mut parent, edge.target().index());
    if a != b {
      parent[a] = b;
    }
  }

  let mut comps: HashMap<usize, Vec<usize>> = HashMap::new();
  for i in 0..n {
    let root = find(&mut parent, i);
    comps.entry(root).or_default().push(i);
  }

  let mut result: Vec<Vec<usize>> = comps.into_values().collect();
  // Sort deterministically: largest first, ties broken by first node index.
  result.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a[0].cmp(&b[0])));
  result
}

/// Lay out a single component in a local [-1, 1]^2 coordinate system.
/// Small components (≤ 8 nodes) use a circular layout; larger ones use
/// a Fruchterman-Reingold force-directed layout with deterministic
/// initialization so the output is reproducible.
fn layout_component(
  graph: &DiGraph<usize, RenderEdgeData>,
  comp: &[usize],
) -> Vec<(f64, f64)> {
  let m = comp.len();
  if m == 0 {
    return vec![];
  }
  if m == 1 {
    return vec![(0.0, 0.0)];
  }
  if m <= 8 {
    return (0..m)
      .map(|k| {
        let angle = PI / 2.0 + (k as f64) * 2.0 * PI / (m as f64);
        (angle.cos(), angle.sin())
      })
      .collect();
  }

  // Map global node index → local index within this component.
  let idx_in_comp: HashMap<usize, usize> = comp
    .iter()
    .enumerate()
    .map(|(i, &node)| (node, i))
    .collect();

  // Undirected adjacency list for the component.
  let mut adj: Vec<Vec<usize>> = vec![Vec::new(); m];
  for edge in graph.edge_references() {
    let si = edge.source().index();
    let di = edge.target().index();
    if let (Some(&a), Some(&b)) = (idx_in_comp.get(&si), idx_in_comp.get(&di))
      && a != b
    {
      adj[a].push(b);
      adj[b].push(a);
    }
  }

  // Deterministic initial placement on a sunflower spiral.
  let golden_angle = PI * (3.0 - (5.0_f64).sqrt());
  let mut pos: Vec<(f64, f64)> = (0..m)
    .map(|i| {
      let r = ((i as f64) + 0.5).sqrt() / (m as f64).sqrt();
      let angle = (i as f64) * golden_angle;
      (r * angle.cos(), r * angle.sin())
    })
    .collect();

  // Fruchterman-Reingold on the unit square [-1, 1]^2.
  let w = 2.0_f64;
  let h = 2.0_f64;
  let area = w * h;
  let k = (area / (m as f64)).sqrt();
  let iterations = 120;
  let mut temp = w * 0.1;
  let cool = (0.01_f64 / temp).powf(1.0 / iterations as f64);

  for _ in 0..iterations {
    let mut disp: Vec<(f64, f64)> = vec![(0.0, 0.0); m];

    // Repulsive force between every pair.
    for i in 0..m {
      for j in (i + 1)..m {
        let dx = pos[i].0 - pos[j].0;
        let dy = pos[i].1 - pos[j].1;
        let d2 = dx * dx + dy * dy;
        let d = d2.sqrt().max(1e-4);
        let force = (k * k) / d;
        let fx = dx / d * force;
        let fy = dy / d * force;
        disp[i].0 += fx;
        disp[i].1 += fy;
        disp[j].0 -= fx;
        disp[j].1 -= fy;
      }
    }

    // Attractive force along edges (each undirected edge is listed twice
    // in `adj`, so we guard on i < j to count it once).
    for i in 0..m {
      for &j in &adj[i] {
        if i >= j {
          continue;
        }
        let dx = pos[i].0 - pos[j].0;
        let dy = pos[i].1 - pos[j].1;
        let d = (dx * dx + dy * dy).sqrt().max(1e-4);
        let force = (d * d) / k;
        let fx = dx / d * force;
        let fy = dy / d * force;
        disp[i].0 -= fx;
        disp[i].1 -= fy;
        disp[j].0 += fx;
        disp[j].1 += fy;
      }
    }

    // Apply displacement, clamped by the current temperature, and keep
    // everything inside [-w/2, w/2] × [-h/2, h/2].
    for i in 0..m {
      let dlen = (disp[i].0 * disp[i].0 + disp[i].1 * disp[i].1)
        .sqrt()
        .max(1e-9);
      let step = dlen.min(temp);
      pos[i].0 += disp[i].0 / dlen * step;
      pos[i].1 += disp[i].1 / dlen * step;
      pos[i].0 = pos[i].0.clamp(-w / 2.0, w / 2.0);
      pos[i].1 = pos[i].1.clamp(-h / 2.0, h / 2.0);
    }

    temp *= cool;
  }

  pos
}

/// Pack per-component layouts into a global [-1, 1]^2 grid. Cells in the
/// grid are weighted roughly by sqrt(component_size) so a 34-node cluster
/// visually dominates a 2-node cluster, but tiny components still get a
/// visible amount of space.
fn pack_components(
  components: &[Vec<usize>],
  component_positions: &[Vec<(f64, f64)>],
  n: usize,
) -> Vec<(f64, f64)> {
  let mut result = vec![(0.0, 0.0); n];
  let k = components.len();
  if k == 0 {
    return result;
  }

  // Arrange cells in a near-square grid. For k = 8 this gives cols = 3.
  let cols = (k as f64).sqrt().ceil() as usize;
  let rows = k.div_ceil(cols);

  let cell_w = 2.0 / cols as f64;
  let cell_h = 2.0 / rows as f64;

  for (ci, (comp, ps)) in components
    .iter()
    .zip(component_positions.iter())
    .enumerate()
  {
    let row = ci / cols;
    let col = ci % cols;

    // Center of this cell in global coords, shifting rows so the grid is
    // centered around (0, 0) and rows go top→bottom (matches usual reading
    // order once the y-axis is interpreted as math-up).
    let cell_cx = -1.0 + (col as f64 + 0.5) * cell_w;
    let cell_cy = 1.0 - (row as f64 + 0.5) * cell_h;

    // Bounding box of this component's local layout.
    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(x, y) in ps {
      if x < min_x {
        min_x = x;
      }
      if x > max_x {
        max_x = x;
      }
      if y < min_y {
        min_y = y;
      }
      if y > max_y {
        max_y = y;
      }
    }
    let local_cx = (min_x + max_x) / 2.0;
    let local_cy = (min_y + max_y) / 2.0;
    let local_w = (max_x - min_x).max(1e-6);
    let local_h = (max_y - min_y).max(1e-6);

    // Fit the component into ~70% of its cell (leaves a visible gutter
    // between clusters), then scale by sqrt(size)/sqrt(max_size) so
    // larger clusters look larger while tiny ones still get a minimum
    // size so single-node components don't shrink to nothing.
    let max_size = components.iter().map(|c| c.len()).max().unwrap_or(1);
    let size_scale =
      ((comp.len() as f64).sqrt() / (max_size as f64).sqrt()).max(0.35);
    let fit_scale = (cell_w * 0.7 / local_w).min(cell_h * 0.7 / local_h);
    let scale = fit_scale * size_scale;

    for (i, &node_idx) in comp.iter().enumerate() {
      let (px, py) = ps[i];
      result[node_idx] = (
        cell_cx + (px - local_cx) * scale,
        cell_cy + (py - local_cy) * scale,
      );
    }
  }

  result
}

fn snap_coord(v: f64) -> f64 {
  for &target in &[0.0, 0.5, -0.5, 1.0, -1.0] {
    if (v - target).abs() < 1e-14 {
      return target;
    }
  }
  v
}

/// Peel off `Labeled[..., label]` and `Style[..., directives...]` wrappers,
/// in any nesting order, returning the innermost expression along with the
/// first label and first color directive encountered.
fn unwrap_edge_wrappers(
  mut expr: &Expr,
) -> (&Expr, Option<String>, Option<Color>) {
  let mut label: Option<String> = None;
  let mut color: Option<Color> = None;
  loop {
    match expr {
      Expr::FunctionCall { name, args }
        if name == "Labeled" && args.len() == 2 =>
      {
        if label.is_none() {
          label = Some(match &args[1] {
            Expr::String(s) => s.clone(),
            other => expr_to_output(other),
          });
        }
        expr = &args[0];
      }
      Expr::FunctionCall { name, args }
        if name == "Style" && args.len() >= 2 =>
      {
        if color.is_none() {
          color = args[1..].iter().find_map(parse_color);
        }
        expr = &args[0];
      }
      _ => break,
    }
  }
  (expr, label, color)
}

/// Peel off `Style[..., directives...]` wrappers from a vertex, returning
/// the innermost expression along with the first color directive.
fn unwrap_vertex_style(mut expr: &Expr) -> (&Expr, Option<Color>) {
  let mut color: Option<Color> = None;
  while let Expr::FunctionCall { name, args } = expr {
    if name == "Style" && args.len() >= 2 {
      if color.is_none() {
        color = args[1..].iter().find_map(parse_color);
      }
      expr = &args[0];
    } else {
      break;
    }
  }
  (expr, color)
}

// ---------------------------------------------------------------------------
// FindCycle
// ---------------------------------------------------------------------------

/// A directed arc derived from an input edge. `edge_id` is shared between the
/// two arcs that come from a single undirected edge so a cycle can never reuse
/// the same undirected edge twice. `directed` records whether the originating
/// edge was directed (controls the head used when rendering the result).
#[derive(Clone)]
struct Arc {
  dst: usize,
  edge_id: usize,
  directed: bool,
}

/// Extract the innermost edge expression, peeling Labeled/Style wrappers.
fn fc_unwrap_edge(e: &Expr) -> &Expr {
  unwrap_edge_wrappers(e).0
}

/// Extract an edge's endpoints and directedness from an edge expression
/// (`u <-> v`, `u -> v`, `UndirectedEdge[u, v]`, `DirectedEdge[u, v]`, possibly
/// wrapped in `Labeled`). Returns `(u, v, directed)`.
pub fn edge_endpoints(e: &Expr) -> Option<(Expr, Expr, bool)> {
  match fc_unwrap_edge(e) {
    Expr::FunctionCall { name, args } if args.len() == 2 => {
      let directed = name != "UndirectedEdge" && name != "TwoWayRule";
      Some((args[0].clone(), args[1].clone(), directed))
    }
    Expr::Rule {
      pattern,
      replacement,
    } => Some((pattern.as_ref().clone(), replacement.as_ref().clone(), true)),
    _ => None,
  }
}

/// EdgeBetweennessCentrality: for each edge, the number of shortest paths (over
/// all ordered vertex pairs) that traverse it, computed by Brandes' algorithm
/// accumulating dependency onto edges. `edge_pairs` are the (i, j) endpoint
/// indices of each edge in list order; the returned values follow that order.
pub fn edge_betweenness_centrality(
  n: usize,
  edge_pairs: &[(usize, usize)],
) -> Vec<f64> {
  use std::collections::{HashMap, VecDeque};
  let m = edge_pairs.len();
  let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
  let mut edge_idx: HashMap<(usize, usize), usize> = HashMap::new();
  for (k, &(i, j)) in edge_pairs.iter().enumerate() {
    adj[i].push(j);
    if i != j {
      adj[j].push(i);
    }
    let key = if i <= j { (i, j) } else { (j, i) };
    edge_idx.entry(key).or_insert(k);
  }
  let mut eb = vec![0.0_f64; m];
  for s in 0..n {
    let mut dist = vec![-1i64; n];
    let mut sigma = vec![0.0_f64; n];
    let mut pred: Vec<Vec<usize>> = vec![vec![]; n];
    let mut stack = Vec::new();
    let mut queue = VecDeque::new();
    dist[s] = 0;
    sigma[s] = 1.0;
    queue.push_back(s);
    while let Some(v) = queue.pop_front() {
      stack.push(v);
      for &w in &adj[v] {
        if dist[w] < 0 {
          dist[w] = dist[v] + 1;
          queue.push_back(w);
        }
        if dist[w] == dist[v] + 1 {
          sigma[w] += sigma[v];
          pred[w].push(v);
        }
      }
    }
    let mut delta = vec![0.0_f64; n];
    while let Some(w) = stack.pop() {
      for &v in &pred[w] {
        let c = (sigma[v] / sigma[w]) * (1.0 + delta[w]);
        let key = if v <= w { (v, w) } else { (w, v) };
        if let Some(&k) = edge_idx.get(&key) {
          eb[k] += c;
        }
        delta[v] += c;
      }
    }
  }
  eb
}

/// Solve the dense linear system `m x = b` by Gaussian elimination with partial
/// pivoting. Returns `None` if the system is singular.
fn solve_linear_f64(mut m: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
  let n = m.len();
  for col in 0..n {
    let mut piv = col;
    for r in (col + 1)..n {
      if m[r][col].abs() > m[piv][col].abs() {
        piv = r;
      }
    }
    if m[piv][col].abs() < 1e-15 {
      return None;
    }
    m.swap(col, piv);
    b.swap(col, piv);
    let d = m[col][col];
    for r in (col + 1)..n {
      let f = m[r][col] / d;
      for c in col..n {
        m[r][c] -= f * m[col][c];
      }
      b[r] -= f * b[col];
    }
  }
  let mut x = vec![0.0_f64; n];
  for i in (0..n).rev() {
    let mut s = b[i];
    for c in (i + 1)..n {
      s -= m[i][c] * x[c];
    }
    x[i] = s / m[i][i];
  }
  Some(x)
}

/// PageRankCentrality: solves (I - alpha P^T) p = (1 - alpha)/n * 1, where
/// P = D^(-1) A is the row-stochastic transition matrix (a zero-out-degree
/// vertex teleports uniformly). The result sums to 1.
pub fn pagerank_centrality(adj: &[Vec<f64>], alpha: f64) -> Option<Vec<f64>> {
  let n = adj.len();
  if n == 0 {
    return Some(vec![]);
  }
  let outdeg: Vec<f64> =
    adj.iter().map(|row| row.iter().sum::<f64>()).collect();
  let mut sys = vec![vec![0.0_f64; n]; n];
  for i in 0..n {
    for j in 0..n {
      let pij = if outdeg[i] > 0.0 {
        adj[i][j] / outdeg[i]
      } else {
        1.0 / n as f64
      };
      // (I - alpha P^T)[j][i] = delta_ij - alpha * P[i][j]
      sys[j][i] = (if i == j { 1.0 } else { 0.0 }) - alpha * pij;
    }
  }
  let b = vec![(1.0 - alpha) / n as f64; n];
  solve_linear_f64(sys, b)
}

/// KatzCentrality: solves (I - alpha A) x = beta * 1 for x, the Katz centrality
/// vector x_i = alpha Σ_j A_ij x_j + beta. Uses Gaussian elimination with
/// partial pivoting; returns `None` if the system is singular.
pub fn katz_centrality(
  adj: &[Vec<f64>],
  alpha: f64,
  beta: f64,
) -> Option<Vec<f64>> {
  let n = adj.len();
  if n == 0 {
    return Some(vec![]);
  }
  // Augmented matrix M = I - alpha A and right-hand side b = beta * 1.
  let mut m = vec![vec![0.0_f64; n]; n];
  for (i, row) in m.iter_mut().enumerate() {
    for (j, mij) in row.iter_mut().enumerate() {
      *mij = (if i == j { 1.0 } else { 0.0 }) - alpha * adj[i][j];
    }
  }
  let mut b = vec![beta; n];
  for col in 0..n {
    let mut piv = col;
    for r in (col + 1)..n {
      if m[r][col].abs() > m[piv][col].abs() {
        piv = r;
      }
    }
    if m[piv][col].abs() < 1e-15 {
      return None; // singular
    }
    m.swap(col, piv);
    b.swap(col, piv);
    let d = m[col][col];
    for r in (col + 1)..n {
      let f = m[r][col] / d;
      for c in col..n {
        m[r][c] -= f * m[col][c];
      }
      b[r] -= f * b[col];
    }
  }
  let mut x = vec![0.0_f64; n];
  for i in (0..n).rev() {
    let mut s = b[i];
    for c in (i + 1)..n {
      s -= m[i][c] * x[c];
    }
    x[i] = s / m[i][i];
  }
  Some(x)
}

/// Perron eigenvalue and (sum-1 normalized, non-negative) eigenvector of an
/// irreducible non-negative matrix, via the same shifted power iteration used
/// by `eigenvector_centrality`. Used for the strongly-connected components of
/// a directed graph.
pub fn perron_eigenpair(m: &[Vec<f64>]) -> (f64, Vec<f64>) {
  let n = m.len();
  if n == 0 {
    return (0.0, vec![]);
  }
  if n == 1 {
    return (m[0][0], vec![1.0]);
  }
  let maxrow = m
    .iter()
    .map(|row| row.iter().sum::<f64>())
    .fold(0.0_f64, f64::max);
  let c = maxrow + 1.0;
  let mut v = vec![1.0 / n as f64; n];
  for _ in 0..100_000 {
    let mut w = vec![0.0_f64; n];
    for (i, wi) in w.iter_mut().enumerate() {
      let mut s = c * v[i];
      for (j, vj) in v.iter().enumerate() {
        s += m[i][j] * vj;
      }
      *wi = s;
    }
    let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
      break;
    }
    for x in w.iter_mut() {
      *x /= norm;
    }
    let diff: f64 = v.iter().zip(&w).map(|(a, b)| (a - b).abs()).sum();
    v = w;
    if diff < 1e-14 {
      break;
    }
  }
  // Rayleigh quotient with the unit-norm Perron vector gives M's eigenvalue.
  let mut mv = vec![0.0_f64; n];
  for (i, mvi) in mv.iter_mut().enumerate() {
    for (j, vj) in v.iter().enumerate() {
      *mvi += m[i][j] * vj;
    }
  }
  let lambda: f64 = (0..n).map(|i| v[i] * mv[i]).sum();
  if v.iter().sum::<f64>() < 0.0 {
    for x in v.iter_mut() {
      *x = -*x;
    }
  }
  let sum: f64 = v.iter().sum();
  if sum != 0.0 {
    for x in v.iter_mut() {
      *x /= sum;
    }
  }
  (lambda, v)
}

/// EigenvectorCentrality: the principal (Perron) eigenvector of the adjacency
/// matrix, made non-negative and normalized so the entries sum to 1. The power
/// iteration uses a spectral shift `A + c I` (c > max degree) so that the
/// dominant eigenvalue is unique even for bipartite graphs (whose adjacency
/// spectrum is symmetric about 0 and would otherwise make plain power iteration
/// oscillate).
pub fn eigenvector_centrality(adj: &[Vec<f64>]) -> Vec<f64> {
  let n = adj.len();
  if n == 0 {
    return vec![];
  }
  if n == 1 {
    return vec![1.0];
  }
  let maxdeg = adj
    .iter()
    .map(|row| row.iter().sum::<f64>())
    .fold(0.0_f64, f64::max);
  let c = maxdeg + 1.0;
  let mut v = vec![1.0 / n as f64; n];
  for _ in 0..100_000 {
    let mut w = vec![0.0_f64; n];
    for (i, wi) in w.iter_mut().enumerate() {
      let mut s = c * v[i];
      for (j, vj) in v.iter().enumerate() {
        s += adj[i][j] * vj;
      }
      *wi = s;
    }
    let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
      break;
    }
    for x in w.iter_mut() {
      *x /= norm;
    }
    let diff: f64 = v.iter().zip(&w).map(|(a, b)| (a - b).abs()).sum();
    v = w;
    if diff < 1e-14 {
      break;
    }
  }
  // Perron eigenvector is single-signed; orient it non-negative.
  if v.iter().sum::<f64>() < 0.0 {
    for x in v.iter_mut() {
      *x = -*x;
    }
  }
  let sum: f64 = v.iter().sum();
  if sum != 0.0 {
    for x in v.iter_mut() {
      *x /= sum;
    }
  }
  v
}

/// GraphReciprocity: the fraction of directed edges that are reciprocated (have
/// a reverse edge); self-loops count as reciprocated. An undirected graph with
/// at least one edge has reciprocity 1. Returns `None` (leave unevaluated) for
/// an edgeless graph, a mixed directed/undirected graph, or a multigraph, all
/// of which wolframscript declines to evaluate.
pub fn graph_reciprocity(edges: &[Expr]) -> Option<Expr> {
  use std::collections::HashSet;
  let parsed: Vec<(Expr, Expr, bool)> =
    edges.iter().filter_map(edge_endpoints).collect();
  if parsed.is_empty() || parsed.len() != edges.len() {
    return None; // no edges, or an unparseable edge
  }
  let has_directed = parsed.iter().any(|(_, _, d)| *d);
  let has_undirected = parsed.iter().any(|(_, _, d)| !*d);
  if has_directed && has_undirected {
    return None; // mixed graph
  }
  let key = |u: &Expr, v: &Expr, directed: bool| -> (String, String) {
    let (ku, kv) = (
      crate::syntax::expr_to_string(u),
      crate::syntax::expr_to_string(v),
    );
    if directed || ku <= kv {
      (ku, kv)
    } else {
      (kv, ku)
    }
  };
  let keys: Vec<(String, String)> =
    parsed.iter().map(|(u, v, d)| key(u, v, *d)).collect();
  if keys.iter().collect::<HashSet<_>>().len() != keys.len() {
    return None; // multigraph
  }
  if has_undirected {
    return Some(Expr::Integer(1));
  }
  let dir_set: HashSet<(String, String)> = keys.iter().cloned().collect();
  let reciprocated = parsed
    .iter()
    .filter(|(u, v, _)| {
      let (ku, kv) = (
        crate::syntax::expr_to_string(u),
        crate::syntax::expr_to_string(v),
      );
      ku == kv || dir_set.contains(&(kv, ku))
    })
    .count();
  Some(crate::functions::math_ast::make_rational(
    reciprocated as i128,
    parsed.len() as i128,
  ))
}

/// Build the disjoint union of several graphs: every vertex is relabeled to a
/// consecutive integer 1..N (the first graph keeps 1..n1, the next is shifted by
/// n1, and so on), and the edges are relabeled to match while preserving their
/// direction. `graphs` is a slice of (vertices, edges) pairs.
pub fn graph_disjoint_union(graphs: &[(&[Expr], &[Expr])]) -> Expr {
  use std::collections::HashMap;
  let mut new_vertices: Vec<Expr> = Vec::new();
  let mut new_edges: Vec<Expr> = Vec::new();
  let mut offset = 0usize;
  for (verts, edges) in graphs {
    let mut label: HashMap<String, i128> = HashMap::new();
    for (j, v) in verts.iter().enumerate() {
      let new_id = (offset + j + 1) as i128;
      label.insert(crate::syntax::expr_to_string(v), new_id);
      new_vertices.push(Expr::Integer(new_id));
    }
    for e in edges.iter() {
      match edge_endpoints(e) {
        Some((u, v, directed)) => {
          match (
            label.get(&crate::syntax::expr_to_string(&u)),
            label.get(&crate::syntax::expr_to_string(&v)),
          ) {
            (Some(&nu), Some(&nv)) => {
              // Undirected endpoints are stored canonically as (min, max).
              let (a, b) = if !directed && nu > nv {
                (nv, nu)
              } else {
                (nu, nv)
              };
              new_edges.push(Expr::FunctionCall {
                name: if directed {
                  "DirectedEdge"
                } else {
                  "UndirectedEdge"
                }
                .to_string(),
                args: vec![Expr::Integer(a), Expr::Integer(b)].into(),
              });
            }
            _ => new_edges.push(e.clone()),
          }
        }
        None => new_edges.push(e.clone()),
      }
    }
    offset += verts.len();
  }
  Expr::FunctionCall {
    name: "Graph".to_string(),
    args: vec![
      Expr::List(new_vertices.into()),
      Expr::List(new_edges.into()),
    ]
    .into(),
  }
}

/// Merge the given vertices of a graph into the first one (`to_contract[0]`),
/// redirecting edges to that survivor and dropping the resulting self-loops and
/// duplicate edges, then re-canonicalizing the edge order (smaller endpoint
/// first, then the larger) the way wolframscript does. Edge direction is
/// preserved. `gargs` are the arguments of the `Graph[vertices, edges, opts...]`
/// expression. Returns the new graph, or `None` if any listed vertex is absent
/// (callers then leave the graph unchanged).
pub fn contract_vertices_in_graph(
  gargs: &[Expr],
  to_contract: &[Expr],
) -> Option<Expr> {
  use crate::evaluator::pattern_matching::expr_equal;
  use crate::functions::list_helpers_ast::sorting::canonical_cmp;
  if gargs.len() < 2 {
    return None;
  }
  let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1]) else {
    return None;
  };
  let valid = !to_contract.is_empty()
    && to_contract
      .iter()
      .all(|d| vertices.iter().any(|v| expr_equal(v, d)));
  if !valid {
    return None;
  }
  let survivor = to_contract[0].clone();
  let remap = |v: &Expr| -> Expr {
    if to_contract.iter().any(|d| expr_equal(v, d)) {
      survivor.clone()
    } else {
      v.clone()
    }
  };
  let is_directed =
    |head: &str| head != "UndirectedEdge" && head != "TwoWayRule";
  // Edge head + endpoints (Labeled-aware).
  fn edge_parts(e: &Expr) -> Option<(String, Expr, Expr)> {
    let inner = match e {
      Expr::FunctionCall { name, args }
        if name == "Labeled" && args.len() == 2 =>
      {
        &args[0]
      }
      _ => e,
    };
    match inner {
      Expr::FunctionCall { name, args } if args.len() == 2 => {
        Some((name.clone(), args[0].clone(), args[1].clone()))
      }
      Expr::Rule {
        pattern,
        replacement,
      } => Some((
        "Rule".to_string(),
        pattern.as_ref().clone(),
        replacement.as_ref().clone(),
      )),
      _ => None,
    }
  }

  let new_vertices: Vec<Expr> = vertices
    .iter()
    .filter(|v| {
      expr_equal(v, &survivor) || !to_contract.iter().any(|d| expr_equal(v, d))
    })
    .cloned()
    .collect();

  let mut new_edges: Vec<Expr> = Vec::new();
  for e in edges.iter() {
    let (head, a, b) = match edge_parts(e) {
      Some(t) => t,
      None => {
        new_edges.push(e.clone());
        continue;
      }
    };
    let directed = is_directed(&head);
    // Undirected endpoints are stored canonically as (min, max).
    let (na, nb) = {
      let (ra, rb) = (remap(&a), remap(&b));
      if !directed && canonical_cmp(&ra, &rb) == std::cmp::Ordering::Greater {
        (rb, ra)
      } else {
        (ra, rb)
      }
    };
    if expr_equal(&na, &nb) {
      continue; // self-loop
    }
    let dup = new_edges.iter().any(|ex| match edge_parts(ex) {
      Some((eh, ea, eb)) if is_directed(&eh) == directed => {
        if directed {
          expr_equal(&ea, &na) && expr_equal(&eb, &nb)
        } else {
          (expr_equal(&ea, &na) && expr_equal(&eb, &nb))
            || (expr_equal(&ea, &nb) && expr_equal(&eb, &na))
        }
      }
      _ => false,
    });
    if dup {
      continue;
    }
    new_edges.push(match e {
      Expr::Rule { .. } => Expr::Rule {
        pattern: Box::new(na),
        replacement: Box::new(nb),
      },
      _ => Expr::FunctionCall {
        name: head.clone(),
        args: vec![na, nb].into(),
      },
    });
  }
  // Re-canonicalize the edge order by endpoints.
  new_edges.sort_by(|e1, e2| match (edge_parts(e1), edge_parts(e2)) {
    (Some((_, a1, b1)), Some((_, a2, b2))) => {
      canonical_cmp(&a1, &a2).then_with(|| canonical_cmp(&b1, &b2))
    }
    _ => std::cmp::Ordering::Equal,
  });
  let mut result_args = vec![
    Expr::List(new_vertices.into()),
    Expr::List(new_edges.into()),
  ];
  result_args.extend(gargs[2..].iter().cloned());
  Some(Expr::FunctionCall {
    name: "Graph".to_string(),
    args: result_args.into(),
  })
}

/// Parse the FindCycle input (first argument) into a vertex list (in first-
/// appearance order, matching VertexList) and the raw edge expressions.
/// Accepts either `Graph[{verts}, {edges}]` or a bare list of edges/rules.
fn fc_parse_input(arg: &Expr) -> Option<(Vec<Expr>, Vec<Expr>)> {
  let raw_edges: Vec<Expr> = match arg {
    Expr::FunctionCall { name, args } if name == "Graph" && args.len() >= 2 => {
      match (&args[0], &args[1]) {
        (Expr::List(_v), Expr::List(e)) => e.iter().cloned().collect(),
        _ => return None,
      }
    }
    Expr::List(edges) => edges.iter().cloned().collect(),
    _ => return None,
  };

  // If the input was a Graph with an explicit vertex list, honour that order.
  if let Expr::FunctionCall { name, args } = arg
    && name == "Graph"
    && args.len() >= 2
    && let Expr::List(v) = &args[0]
  {
    let verts: Vec<Expr> = v.iter().cloned().collect();
    if !verts.is_empty() {
      return Some((verts, raw_edges));
    }
  }

  // Otherwise derive the vertex list from edge endpoints, in first-appearance
  // order (this matches wolframscript's VertexList of an edge list).
  let mut seen: Vec<String> = Vec::new();
  let mut verts: Vec<Expr> = Vec::new();
  let push = |v: &Expr, seen: &mut Vec<String>, verts: &mut Vec<Expr>| {
    let key = expr_to_string(v);
    if !seen.contains(&key) {
      seen.push(key);
      verts.push(v.clone());
    }
  };
  for e in &raw_edges {
    let inner = fc_unwrap_edge(e);
    match inner {
      Expr::FunctionCall { args: eargs, .. } if eargs.len() == 2 => {
        push(&eargs[0], &mut seen, &mut verts);
        push(&eargs[1], &mut seen, &mut verts);
      }
      Expr::Rule {
        pattern,
        replacement,
      } => {
        push(pattern, &mut seen, &mut verts);
        push(replacement, &mut seen, &mut verts);
      }
      _ => {}
    }
  }
  Some((verts, raw_edges))
}

/// Parse the (optional) kspec second argument into an inclusive length range.
/// Returns (kmin, kmax). `Infinity` (the default) maps to usize::MAX.
fn fc_parse_kspec(arg: Option<&Expr>) -> Option<(usize, usize)> {
  match arg {
    None => Some((1, usize::MAX)),
    Some(Expr::Identifier(s)) if s == "Infinity" => Some((1, usize::MAX)),
    Some(Expr::Integer(k)) if *k >= 1 => Some((1, *k as usize)),
    Some(Expr::List(items)) => match items.len() {
      1 => match &items[0] {
        Expr::Integer(k) if *k >= 1 => Some((*k as usize, *k as usize)),
        _ => None,
      },
      2 => {
        let lo = match &items[0] {
          Expr::Integer(k) if *k >= 1 => *k as usize,
          _ => return None,
        };
        let hi = match &items[1] {
          Expr::Integer(k) if *k >= 1 => *k as usize,
          Expr::Identifier(s) if s == "Infinity" => usize::MAX,
          _ => return None,
        };
        Some((lo, hi))
      }
      _ => None,
    },
    _ => None,
  }
}

/// Parse the (optional) count third argument: number of cycles to return.
/// `All` maps to usize::MAX, default (None) is 1.
fn fc_parse_count(arg: Option<&Expr>) -> Option<usize> {
  match arg {
    None => Some(1),
    Some(Expr::Identifier(s)) if s == "All" || s == "Infinity" => {
      Some(usize::MAX)
    }
    Some(Expr::Integer(n)) if *n >= 0 => Some(*n as usize),
    _ => None,
  }
}

/// Core FindCycle implementation. Returns a list of cycles, each cycle being a
/// list of edge expressions (DirectedEdge / UndirectedEdge).
pub fn find_cycle_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "FindCycle".to_string(),
      args: args.to_vec().into(),
    })
  };

  if args.is_empty() || args.len() > 3 {
    return unevaluated();
  }

  let (vertices, raw_edges) = match fc_parse_input(&args[0]) {
    Some(p) => p,
    None => return unevaluated(),
  };
  let (kmin, kmax) = match fc_parse_kspec(args.get(1)) {
    Some(k) => k,
    None => return unevaluated(),
  };
  let max_count = match fc_parse_count(args.get(2)) {
    Some(n) => n,
    None => return unevaluated(),
  };

  let n = vertices.len();
  if n == 0 || max_count == 0 {
    return Ok(Expr::List(vec![].into()));
  }

  // Map vertex key -> rank (index in vertex list).
  let mut rank: HashMap<String, usize> = HashMap::new();
  for (i, v) in vertices.iter().enumerate() {
    rank.entry(expr_to_string(v)).or_insert(i);
  }

  // Build the arc adjacency list. Each input edge gets a unique edge_id; an
  // undirected edge contributes two arcs that share that id.
  let mut adj: Vec<Vec<Arc>> = vec![Vec::new(); n];
  let mut next_edge_id = 0usize;
  for e in &raw_edges {
    let inner = fc_unwrap_edge(e);
    let (src, dst, directed) = match inner {
      Expr::FunctionCall { name, args: eargs } if eargs.len() == 2 => {
        let d = name != "UndirectedEdge" && name != "TwoWayRule";
        (&eargs[0], &eargs[1], d)
      }
      Expr::Rule {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref(), true),
      _ => continue,
    };
    let (si, di) = match (
      rank.get(&expr_to_string(src)),
      rank.get(&expr_to_string(dst)),
    ) {
      (Some(&s), Some(&d)) => (s, d),
      _ => continue,
    };
    let id = next_edge_id;
    next_edge_id += 1;
    adj[si].push(Arc {
      dst: di,
      edge_id: id,
      directed,
    });
    if !directed {
      adj[di].push(Arc {
        dst: si,
        edge_id: id,
        directed,
      });
    }
  }

  // Render an edge (src, dst, directed) as the appropriate Expr.
  let render_edge = |s: usize, d: usize, directed: bool| -> Expr {
    Expr::FunctionCall {
      name: if directed {
        "DirectedEdge".to_string()
      } else {
        "UndirectedEdge".to_string()
      },
      args: vec![vertices[s].clone(), vertices[d].clone()].into(),
    }
  };

  // Collect cycles in DFS discovery order. Each cycle is rooted at its minimum
  // rank vertex (the DFS root), and only explores vertices of rank >= root so
  // each undirected/directed simple cycle is enumerated exactly once.
  // Each entry carries (root_rank, edges).
  let mut cycles: Vec<(usize, Vec<Expr>)> = Vec::new();
  // path of (src_vertex, arc) describing edges taken so far.
  let mut path_edges: Vec<(usize, Arc)> = Vec::new();
  let mut on_path = vec![false; n];
  let mut used_edge: HashMap<usize, bool> = HashMap::new();
  // Sorted edge-id sets of undirected cycles already emitted, used to avoid
  // reporting the same undirected cycle in its reverse orientation.
  let mut seen_undirected: std::collections::HashSet<Vec<usize>> =
    std::collections::HashSet::new();

  // We stop early once we have collected enough cycles. For the default count
  // (1) and small/medium counts this avoids enumerating the whole graph; for
  // All (usize::MAX) we enumerate everything.
  for root in 0..n {
    if cycles.len() >= max_count {
      break;
    }
    // Explicit recursion via a helper closure is awkward in Rust, so use an
    // iterative-style recursive fn defined locally.
    #[allow(clippy::too_many_arguments)]
    fn dfs(
      cur: usize,
      root: usize,
      adj: &[Vec<Arc>],
      on_path: &mut [bool],
      path_edges: &mut Vec<(usize, Arc)>,
      used_edge: &mut HashMap<usize, bool>,
      kmin: usize,
      kmax: usize,
      cycles: &mut Vec<(usize, Vec<Expr>)>,
      seen_undirected: &mut std::collections::HashSet<Vec<usize>>,
      max_count: usize,
      render_edge: &dyn Fn(usize, usize, bool) -> Expr,
    ) {
      if cycles.len() >= max_count {
        return;
      }
      for arc in &adj[cur] {
        if cycles.len() >= max_count {
          return;
        }
        if arc.dst < root {
          continue; // keep root as the minimum-rank vertex of the cycle
        }
        if *used_edge.get(&arc.edge_id).unwrap_or(&false) {
          continue; // don't reuse the same input edge
        }
        if arc.dst == root {
          // Completed a cycle back to the root. A length-1 cycle is a
          // self-loop, which FindCycle does not report.
          let len = path_edges.len() + 1;
          if len >= 2 && len >= kmin && len <= kmax {
            // If every edge of the cycle is undirected, the reverse traversal
            // describes the same cycle. Deduplicate by the (sorted) set of
            // originating edge ids so each undirected cycle is reported once,
            // keeping the orientation that DFS discovers first.
            let all_undirected =
              !arc.directed && path_edges.iter().all(|(_, a)| !a.directed);
            let keep = if all_undirected {
              let mut ids: Vec<usize> =
                path_edges.iter().map(|(_, a)| a.edge_id).collect();
              ids.push(arc.edge_id);
              ids.sort_unstable();
              seen_undirected.insert(ids)
            } else {
              true
            };
            if keep {
              let mut cyc: Vec<Expr> = path_edges
                .iter()
                .map(|(s, a)| render_edge(*s, a.dst, a.directed))
                .collect();
              cyc.push(render_edge(cur, arc.dst, arc.directed));
              cycles.push((root, cyc));
              if cycles.len() >= max_count {
                return;
              }
            }
          }
          continue;
        }
        if on_path[arc.dst] {
          continue; // simple cycle: no repeated intermediate vertex
        }
        if path_edges.len() + 1 >= kmax {
          continue; // can't extend further and still close within kmax
        }
        on_path[arc.dst] = true;
        used_edge.insert(arc.edge_id, true);
        path_edges.push((cur, arc.clone()));
        dfs(
          arc.dst,
          root,
          adj,
          on_path,
          path_edges,
          used_edge,
          kmin,
          kmax,
          cycles,
          seen_undirected,
          max_count,
          render_edge,
        );
        path_edges.pop();
        used_edge.insert(arc.edge_id, false);
        on_path[arc.dst] = false;
      }
    }

    on_path[root] = true;
    dfs(
      root,
      root,
      &adj,
      &mut on_path,
      &mut path_edges,
      &mut used_edge,
      kmin,
      kmax,
      &mut cycles,
      &mut seen_undirected,
      max_count,
      &render_edge,
    );
    on_path[root] = false;
  }

  // Selection happens in DFS-discovery order (already truncated to max_count by
  // the search). wolframscript then orders the returned cycles by length
  // ascending, breaking ties by descending root rank and, finally, by reverse
  // discovery order. Reversing the vector first and using a stable sort yields
  // that reverse-discovery tiebreak for cycles that are otherwise equal.
  cycles.reverse();
  cycles.sort_by(|a, b| a.1.len().cmp(&b.1.len()).then_with(|| b.0.cmp(&a.0)));

  Ok(Expr::List(
    cycles
      .into_iter()
      .map(|(_, c)| Expr::List(c.into()))
      .collect(),
  ))
}

fn collect_directives(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::FunctionCall { name, args } if name == "Directive" => args.to_vec(),
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

/// FindShortestPath[graph, src, dst] — shortest weighted path from `src` to
/// `dst` as a list of vertices (empty list if unreachable). A trailing
/// `Method -> …` option is accepted and ignored (the result is method-
/// independent). Edge weights come from the graph's `EdgeWeight` option
/// (default 1 per edge); undirected edges are traversable both ways.
/// FindFundamentalCycles[g] — fundamental cycles with respect to a BFS
/// spanning forest, matching wolframscript: one cycle per non-tree edge,
/// cycles listed in reverse EdgeList order of their non-tree edge; each
/// cycle walks from the closing edge's second endpoint up to the LCA,
/// back down to the first endpoint, and closes with the non-tree edge in
/// its input orientation. Directed graphs emit `ngen`, non-graphs `graph`,
/// and non-option extra arguments `nonopt` — all like wolframscript.
pub fn find_fundamental_cycles_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "FindFundamentalCycles".to_string(),
      args: args.to_vec().into(),
    })
  };
  let call_display = || {
    expr_to_string(&Expr::FunctionCall {
      name: "FindFundamentalCycles".to_string(),
      args: args.to_vec().into(),
    })
  };

  if args.is_empty() {
    crate::emit_message(
      "FindFundamentalCycles::argx: FindFundamentalCycles called with 0 arguments; 1 argument is expected.",
    );
    return unevaluated();
  }

  // Arguments beyond the graph must be options; they are ignored.
  for extra in &args[1..] {
    let is_opt = matches!(
      extra,
      Expr::Rule { .. } | Expr::RuleDelayed { .. } | Expr::List(_)
    );
    if !is_opt {
      crate::emit_message(&format!(
        "FindFundamentalCycles::nonopt: Options expected (instead of {}) beyond position 1 in {}. An option must be a rule or a list of rules.",
        expr_to_string(extra),
        call_display()
      ));
      return unevaluated();
    }
  }

  let graph_expected = || {
    crate::emit_message(&format!(
      "FindFundamentalCycles::graph: A graph object is expected at position 1 in {}.",
      call_display()
    ));
  };

  let Some((vertices, raw_edges)) = fc_parse_input(&args[0]) else {
    graph_expected();
    return unevaluated();
  };

  let n = vertices.len();
  let mut rank: HashMap<String, usize> = HashMap::new();
  for (i, v) in vertices.iter().enumerate() {
    rank.entry(expr_to_string(v)).or_insert(i);
  }

  // Edge instances in input order, keeping the input orientation.
  let mut edges: Vec<(usize, usize)> = Vec::new();
  for e in &raw_edges {
    let inner = fc_unwrap_edge(e);
    let (src, dst, directed) = match inner {
      Expr::FunctionCall { name, args: eargs } if eargs.len() == 2 => {
        let d = name != "UndirectedEdge" && name != "TwoWayRule";
        (&eargs[0], &eargs[1], d)
      }
      Expr::Rule {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref(), true),
      _ => {
        graph_expected();
        return unevaluated();
      }
    };
    if directed {
      crate::emit_message(&format!(
        "FindFundamentalCycles::ngen: The generalized {} is not implemented.",
        call_display()
      ));
      return unevaluated();
    }
    match (
      rank.get(&expr_to_string(src)),
      rank.get(&expr_to_string(dst)),
    ) {
      (Some(&si), Some(&di)) => edges.push((si, di)),
      _ => {
        graph_expected();
        return unevaluated();
      }
    }
  }

  // BFS spanning forest: roots in vertex order, neighbours in edge input
  // order.
  let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
  for (idx, &(u, v)) in edges.iter().enumerate() {
    adj[u].push((v, idx));
    if u != v {
      adj[v].push((u, idx));
    }
  }
  let mut parent: Vec<Option<usize>> = vec![None; n];
  let mut visited = vec![false; n];
  let mut tree_edge = vec![false; edges.len()];
  for root in 0..n {
    if visited[root] {
      continue;
    }
    visited[root] = true;
    let mut queue = std::collections::VecDeque::from([root]);
    while let Some(u) = queue.pop_front() {
      for &(v, idx) in &adj[u] {
        if !visited[v] {
          visited[v] = true;
          parent[v] = Some(u);
          tree_edge[idx] = true;
          queue.push_back(v);
        }
      }
    }
  }

  let render_edge = |s: usize, d: usize| -> Expr {
    Expr::FunctionCall {
      name: "UndirectedEdge".to_string(),
      args: vec![vertices[s].clone(), vertices[d].clone()].into(),
    }
  };
  let chain_to_root = |mut x: usize| -> Vec<usize> {
    let mut c = vec![x];
    while let Some(p) = parent[x] {
      c.push(p);
      x = p;
    }
    c
  };

  let mut cycles: Vec<Expr> = Vec::new();
  for idx in (0..edges.len()).rev() {
    if tree_edge[idx] {
      continue;
    }
    let (u, v) = edges[idx];
    let mut cycle: Vec<Expr> = Vec::new();
    if u != v {
      let cu = chain_to_root(u);
      let cv = chain_to_root(v);
      // Strip the common tail (below the LCA both chains coincide).
      let mut lu = cu.len();
      let mut lv = cv.len();
      while lu > 1 && lv > 1 && cu[lu - 2] == cv[lv - 2] {
        lu -= 1;
        lv -= 1;
      }
      // cu[lu - 1] == cv[lv - 1] is the LCA.
      for i in 0..lv - 1 {
        cycle.push(render_edge(cv[i], cv[i + 1]));
      }
      for i in (0..lu - 1).rev() {
        cycle.push(render_edge(cu[i + 1], cu[i]));
      }
    }
    cycle.push(render_edge(u, v));
    cycles.push(Expr::List(cycle.into()));
  }

  Ok(Expr::List(cycles.into()))
}

pub fn find_shortest_path_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let inert = || {
    Ok(Expr::FunctionCall {
      name: "FindShortestPath".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() < 3 {
    return inert();
  }

  // Locate the edge list and the EdgeWeight option inside the Graph.
  let graph_args = match &args[0] {
    Expr::FunctionCall { name, args } if name == "Graph" => args.as_slice(),
    _ => return inert(),
  };
  let is_edge = |e: &Expr| {
    let (inner, _, _) = unwrap_edge_wrappers(e);
    matches!(inner, Expr::FunctionCall { name, args }
      if (name == "DirectedEdge" || name == "UndirectedEdge") && args.len() == 2)
      || matches!(e, Expr::Rule { .. })
  };
  let edges: &[Expr] = match graph_args.iter().find_map(|a| match a {
    Expr::List(items) if !items.is_empty() && items.iter().all(is_edge) => {
      Some(items.as_slice())
    }
    _ => None,
  }) {
    Some(e) => e,
    None => return inert(),
  };
  let weights: Vec<f64> = graph_args
    .iter()
    .find_map(|a| match a {
      Expr::Rule { pattern, replacement }
        if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "EdgeWeight") =>
      {
        if let Expr::List(ws) = replacement.as_ref() {
          Some(
            ws.iter()
              .map(|w| {
                crate::functions::math_ast::try_eval_to_f64(w).unwrap_or(1.0)
              })
              .collect::<Vec<_>>(),
          )
        } else {
          None
        }
      }
      _ => None,
    })
    .unwrap_or_default();

  // Build the adjacency list keyed by the vertex string form, remembering the
  // original vertex Expr for each key so the output preserves it.
  let mut adj: HashMap<String, Vec<(String, f64)>> = HashMap::new();
  let mut vexpr: HashMap<String, Expr> = HashMap::new();
  for (i, e) in edges.iter().enumerate() {
    let (inner, _, _) = unwrap_edge_wrappers(e);
    let (s, d, directed) = match inner {
      Expr::FunctionCall { name, args } if args.len() == 2 => {
        (args[0].clone(), args[1].clone(), name == "DirectedEdge")
      }
      _ => match e {
        Expr::Rule {
          pattern,
          replacement,
        } => ((**pattern).clone(), (**replacement).clone(), true),
        _ => continue,
      },
    };
    let w = weights.get(i).copied().unwrap_or(1.0);
    let sk = expr_to_string(&s);
    let dk = expr_to_string(&d);
    vexpr.entry(sk.clone()).or_insert_with(|| s.clone());
    vexpr.entry(dk.clone()).or_insert_with(|| d.clone());
    adj.entry(sk.clone()).or_default().push((dk.clone(), w));
    if !directed {
      adj.entry(dk).or_default().push((sk, w));
    }
  }

  let start = expr_to_string(&args[1]);
  let goal = expr_to_string(&args[2]);
  if !vexpr.contains_key(&start) || !vexpr.contains_key(&goal) {
    return Ok(Expr::List(vec![].into()));
  }

  // Dijkstra (binary min-heap, non-negative costs) returning the shortest
  // distance from `src` to every reachable node over the given adjacency.
  use std::collections::BinaryHeap;
  let dijkstra = |adj: &HashMap<String, Vec<(String, f64)>>,
                  src: &str|
   -> HashMap<String, f64> {
    let mut dist: HashMap<String, f64> = HashMap::new();
    let mut heap: BinaryHeap<(
      std::cmp::Reverse<ordered_f64::OrderedF64>,
      String,
    )> = BinaryHeap::new();
    dist.insert(src.to_string(), 0.0);
    heap.push((
      std::cmp::Reverse(ordered_f64::OrderedF64(0.0)),
      src.to_string(),
    ));
    while let Some((std::cmp::Reverse(ordered_f64::OrderedF64(d)), u)) =
      heap.pop()
    {
      if d > *dist.get(&u).unwrap_or(&f64::INFINITY) {
        continue;
      }
      if let Some(neighbors) = adj.get(&u) {
        for (v, w) in neighbors {
          let nd = d + w;
          if nd < *dist.get(v).unwrap_or(&f64::INFINITY) {
            dist.insert(v.clone(), nd);
            heap.push((
              std::cmp::Reverse(ordered_f64::OrderedF64(nd)),
              v.clone(),
            ));
          }
        }
      }
    }
    dist
  };

  // Reverse adjacency so the distance *to* the goal can be computed; for
  // undirected edges (added both ways above) this equals the forward graph.
  let mut radj: HashMap<String, Vec<(String, f64)>> = HashMap::new();
  for (u, neighbors) in &adj {
    for (v, w) in neighbors {
      radj.entry(v.clone()).or_default().push((u.clone(), *w));
    }
  }
  let dist_to_goal = dijkstra(&radj, &goal);
  if !dist_to_goal.contains_key(&start) {
    return Ok(Expr::List(vec![].into()));
  }

  // Canonical vertex order for tie-breaking: numeric when both keys are
  // numbers (so 2 < 10), otherwise lexicographic on the string form. This
  // reproduces wolframscript, which returns the lexicographically smallest
  // shortest path (the one favouring lower-numbered vertices).
  let vcmp = |a: &str, b: &str| -> std::cmp::Ordering {
    match (a.parse::<i128>(), b.parse::<i128>()) {
      (Ok(x), Ok(y)) => x.cmp(&y),
      _ => match (a.parse::<f64>(), b.parse::<f64>()) {
        (Ok(x), Ok(y)) => {
          x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal)
        }
        _ => a.cmp(b),
      },
    }
  };

  // Greedily walk from start to goal: at each node step to the smallest
  // neighbour that keeps us on a shortest path (w + dist_to_goal[v] equals
  // the node's own distance to the goal).
  let eps = 1e-9;
  let mut path_keys = vec![start.clone()];
  let mut cur = start.clone();
  while cur != goal {
    let dg_cur = match dist_to_goal.get(&cur) {
      Some(d) => *d,
      None => return Ok(Expr::List(vec![].into())),
    };
    let mut best: Option<&String> = None;
    if let Some(neighbors) = adj.get(&cur) {
      for (v, w) in neighbors {
        if let Some(&dgv) = dist_to_goal.get(v)
          && (w + dgv - dg_cur).abs() < eps
        {
          best = match best {
            Some(b) if vcmp(b, v) != std::cmp::Ordering::Greater => Some(b),
            _ => Some(v),
          };
        }
      }
    }
    match best {
      Some(v) => {
        let v = v.clone();
        path_keys.push(v.clone());
        cur = v;
      }
      None => return Ok(Expr::List(vec![].into())),
    }
    // Safety against zero-weight cycles: a simple path can't exceed |V|.
    if path_keys.len() > vexpr.len() {
      return Ok(Expr::List(vec![].into()));
    }
  }
  let path: Vec<Expr> = path_keys.iter().map(|k| vexpr[k].clone()).collect();
  Ok(Expr::List(path.into()))
}

/// Total-order wrapper for f64 Dijkstra costs (all non-negative, finite).
mod ordered_f64 {
  #[derive(PartialEq)]
  pub(super) struct OrderedF64(pub f64);
  impl Eq for OrderedF64 {}
  impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
      Some(self.cmp(other))
    }
  }
  impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
      self
        .0
        .partial_cmp(&other.0)
        .unwrap_or(std::cmp::Ordering::Equal)
    }
  }
}

/// TransitiveClosureGraph[g] - graph with an edge from u to v whenever
/// v is reachable from u in g (no self-loops, matching wolframscript;
/// undirected graphs connect every pair inside a component). Edge
/// lists are accepted by wrapping them in Graph first.
pub fn transitive_closure_graph_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "TransitiveClosureGraph".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 1 {
    return Ok(unevaluated(args));
  }
  // Accept raw edge lists like wolframscript does
  let graph = match &args[0] {
    Expr::List(_) => {
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Graph".to_string(),
        args: vec![args[0].clone()].into(),
      })?
    }
    other => other.clone(),
  };
  let (vertices, edges) = match &graph {
    Expr::FunctionCall { name, args: gargs }
      if name == "Graph" && gargs.len() >= 2 =>
    {
      match (&gargs[0], &gargs[1]) {
        (Expr::List(v), Expr::List(e)) => (v.clone(), e.clone()),
        _ => return Ok(unevaluated(args)),
      }
    }
    _ => return Ok(unevaluated(args)),
  };

  let key = |e: &Expr| crate::syntax::expr_to_string(e);
  let index: std::collections::HashMap<String, usize> = vertices
    .iter()
    .enumerate()
    .map(|(i, v)| (key(v), i))
    .collect();
  let n = vertices.len();
  let mut adjacency = vec![vec![false; n]; n];
  let mut any_directed = false;
  for edge in edges.iter() {
    if let Expr::FunctionCall { name, args: eargs } = edge
      && eargs.len() == 2
      && let (Some(&u), Some(&v)) =
        (index.get(&key(&eargs[0])), index.get(&key(&eargs[1])))
    {
      match name.as_str() {
        "DirectedEdge" => {
          any_directed = true;
          adjacency[u][v] = true;
        }
        "UndirectedEdge" => {
          adjacency[u][v] = true;
          adjacency[v][u] = true;
        }
        _ => return Ok(unevaluated(args)),
      }
    } else {
      return Ok(unevaluated(args));
    }
  }

  // Floyd-Warshall reachability
  let mut reach = adjacency;
  for k in 0..n {
    for i in 0..n {
      if reach[i][k] {
        for j in 0..n {
          if reach[k][j] {
            reach[i][j] = true;
          }
        }
      }
    }
  }

  let edge_head = if any_directed {
    "DirectedEdge"
  } else {
    "UndirectedEdge"
  };
  let mut closure_edges: Vec<Expr> = Vec::new();
  for i in 0..n {
    let j_start = if any_directed { 0 } else { i + 1 };
    for j in j_start..n {
      if i != j && reach[i][j] {
        closure_edges.push(Expr::FunctionCall {
          name: edge_head.to_string(),
          args: vec![vertices[i].clone(), vertices[j].clone()].into(),
        });
      }
    }
  }
  Ok(Expr::FunctionCall {
    name: "Graph".to_string(),
    args: vec![Expr::List(vertices), Expr::List(closure_edges.into())].into(),
  })
}

/// TransitiveReductionGraph[graph | edgeList] → the minimal graph with the
/// same reachability relation. For a directed acyclic graph this reduction is
/// unique and is a subgraph: edge (u, v) is kept iff there is no longer path
/// from u to v. Graphs containing a directed cycle (including any undirected
/// edge or self-loop) are left unevaluated, since Wolfram's reduction there
/// relies on an internal strongly-connected-component convention.
pub fn transitive_reduction_graph_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "TransitiveReductionGraph".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 1 {
    return Ok(unevaluated(args));
  }
  // Accept raw edge lists like wolframscript does.
  let graph = match &args[0] {
    Expr::List(_) => {
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Graph".to_string(),
        args: vec![args[0].clone()].into(),
      })?
    }
    other => other.clone(),
  };
  let (vertices, edges) = match &graph {
    Expr::FunctionCall { name, args: gargs }
      if name == "Graph" && gargs.len() >= 2 =>
    {
      match (&gargs[0], &gargs[1]) {
        (Expr::List(v), Expr::List(e)) => (v.clone(), e.clone()),
        _ => return Ok(unevaluated(args)),
      }
    }
    _ => return Ok(unevaluated(args)),
  };

  let key = |e: &Expr| crate::syntax::expr_to_string(e);
  let index: std::collections::HashMap<String, usize> = vertices
    .iter()
    .enumerate()
    .map(|(i, v)| (key(v), i))
    .collect();
  let n = vertices.len();
  let mut adjacency = vec![vec![false; n]; n];
  for edge in edges.iter() {
    if let Expr::FunctionCall { name, args: eargs } = edge
      && eargs.len() == 2
      && let (Some(&u), Some(&v)) =
        (index.get(&key(&eargs[0])), index.get(&key(&eargs[1])))
    {
      match name.as_str() {
        // An undirected edge behaves like a 2-cycle, which triggers the
        // strongly-connected fallback below.
        "DirectedEdge" => adjacency[u][v] = true,
        "UndirectedEdge" => {
          adjacency[u][v] = true;
          adjacency[v][u] = true;
        }
        _ => return Ok(unevaluated(args)),
      }
    } else {
      return Ok(unevaluated(args));
    }
  }

  // Floyd-Warshall reachability over paths of length >= 1.
  let mut reach = adjacency.clone();
  for k in 0..n {
    for i in 0..n {
      if reach[i][k] {
        for j in 0..n {
          if reach[k][j] {
            reach[i][j] = true;
          }
        }
      }
    }
  }

  // Any vertex that can reach itself lies on a cycle → not a DAG.
  if (0..n).any(|i| reach[i][i]) {
    return Ok(unevaluated(args));
  }

  // Keep a direct edge (i, j) only when no intermediate vertex w provides an
  // alternative path i → w → j (safe for a DAG: such a w cannot use edge
  // (i, j) without creating a cycle).
  let mut reduced_edges: Vec<Expr> = Vec::new();
  for i in 0..n {
    for j in 0..n {
      if !adjacency[i][j] {
        continue;
      }
      let redundant =
        (0..n).any(|w| w != i && w != j && reach[i][w] && reach[w][j]);
      if !redundant {
        reduced_edges.push(Expr::FunctionCall {
          name: "DirectedEdge".to_string(),
          args: vec![vertices[i].clone(), vertices[j].clone()].into(),
        });
      }
    }
  }

  Ok(Expr::FunctionCall {
    name: "Graph".to_string(),
    args: vec![Expr::List(vertices), Expr::List(reduced_edges.into())].into(),
  })
}

/// ReverseGraph[g] - the graph with every directed edge reversed (undirected
/// edges are unchanged). The vertex list is kept as is; the edges come out
/// stably sorted by (source position, target position) in the vertex list —
/// the readout order wolframscript produces from its reversed adjacency
/// structure.
pub fn reverse_graph_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "ReverseGraph".to_string(),
    args: args.to_vec().into(),
  };
  let not_a_graph = || {
    crate::emit_message(&format!(
      "ReverseGraph::graph: A graph object is expected at position 1 in {}.",
      crate::syntax::expr_to_string(&unevaluated())
    ));
    Ok(unevaluated())
  };
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated());
  }
  let (vertices, edges, extra) = match &args[0] {
    Expr::FunctionCall { name, args: gargs }
      if name == "Graph" && gargs.len() >= 2 =>
    {
      match (&gargs[0], &gargs[1]) {
        (Expr::List(v), Expr::List(e)) => {
          (v.clone(), e.clone(), gargs[2..].to_vec())
        }
        _ => return not_a_graph(),
      }
    }
    _ => return not_a_graph(),
  };

  let key = |e: &Expr| crate::syntax::expr_to_string(e);
  let index: std::collections::HashMap<String, usize> = vertices
    .iter()
    .enumerate()
    .map(|(i, v)| (key(v), i))
    .collect();

  // Reverse directed edges in place, then stable-sort everything by the
  // vertex-list positions of the (new) endpoints.
  let mut reversed: Vec<(usize, usize, Expr)> = Vec::with_capacity(edges.len());
  for edge in edges.iter() {
    // `a -> b` edges may still be raw Rule nodes in the explicit
    // Graph[vertices, edges] form.
    let (head, from, to) = match edge {
      Expr::FunctionCall { name, args: eargs } if eargs.len() == 2 => {
        (name.as_str(), &eargs[0], &eargs[1])
      }
      Expr::Rule {
        pattern,
        replacement,
      } => ("Rule", pattern.as_ref(), replacement.as_ref()),
      _ => return not_a_graph(),
    };
    let flipped = match head {
      "DirectedEdge" | "Rule" => Expr::FunctionCall {
        name: "DirectedEdge".to_string(),
        args: vec![to.clone(), from.clone()].into(),
      },
      "UndirectedEdge" => edge.clone(),
      _ => return not_a_graph(),
    };
    let (src, dst) = match &flipped {
      Expr::FunctionCall { args: fargs, .. } => (&fargs[0], &fargs[1]),
      _ => unreachable!(),
    };
    match (index.get(&key(src)), index.get(&key(dst))) {
      (Some(&s), Some(&d)) => reversed.push((s, d, flipped)),
      _ => return not_a_graph(),
    }
  }
  reversed.sort_by_key(|(s, d, _)| (*s, *d));

  let mut new_args = vec![
    Expr::List(vertices),
    Expr::List(reversed.into_iter().map(|(_, _, e)| e).collect()),
  ];
  new_args.extend(extra);
  Ok(Expr::FunctionCall {
    name: "Graph".to_string(),
    args: new_args.into(),
  })
}

/// DirectedGraph[g] - the directed version of `g`. Each undirected edge
/// becomes a pair of opposite directed edges (a self-loop stays a single
/// directed loop); already-directed edges are kept as-is. Edges are ordered by
/// the vertex-list positions of their endpoints, matching wolframscript. The
/// two-argument conversion forms ("Acyclic", "Random", …) are not handled.
pub fn directed_graph_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "DirectedGraph".to_string(),
    args: args.to_vec().into(),
  };
  let not_a_graph = || {
    crate::emit_message(&format!(
      "DirectedGraph::graph: A graph object is expected at position 1 in {}.",
      crate::syntax::expr_to_string(&unevaluated())
    ));
    Ok(unevaluated())
  };
  if args.len() != 1 {
    return Ok(unevaluated());
  }
  let (vertices, edges, extra) = match &args[0] {
    Expr::FunctionCall { name, args: gargs }
      if name == "Graph" && gargs.len() >= 2 =>
    {
      match (&gargs[0], &gargs[1]) {
        (Expr::List(v), Expr::List(e)) => {
          (v.clone(), e.clone(), gargs[2..].to_vec())
        }
        _ => return not_a_graph(),
      }
    }
    _ => return not_a_graph(),
  };

  let key = |e: &Expr| crate::syntax::expr_to_string(e);
  let index: std::collections::HashMap<String, usize> = vertices
    .iter()
    .enumerate()
    .map(|(i, v)| (key(v), i))
    .collect();

  // Emit directed edges, then stable-sort by the vertex-list positions of the
  // endpoints (matching ReverseGraph's ordering convention).
  let mut out: Vec<(usize, usize, Expr)> = Vec::with_capacity(edges.len() * 2);
  for edge in edges.iter() {
    let (head, a, b) = match edge {
      Expr::FunctionCall { name, args: eargs } if eargs.len() == 2 => {
        (name.as_str(), &eargs[0], &eargs[1])
      }
      Expr::Rule {
        pattern,
        replacement,
      } => ("Rule", pattern.as_ref(), replacement.as_ref()),
      _ => return not_a_graph(),
    };
    // The directed orientations contributed by this edge: directed edges keep
    // their single orientation; an undirected edge yields both directions,
    // except a self-loop which stays a single directed loop.
    let orientations: Vec<(&Expr, &Expr)> = match head {
      "DirectedEdge" | "Rule" => vec![(a, b)],
      "UndirectedEdge" if key(a) == key(b) => vec![(a, b)],
      "UndirectedEdge" => vec![(a, b), (b, a)],
      _ => return not_a_graph(),
    };
    for (from, to) in orientations {
      match (index.get(&key(from)), index.get(&key(to))) {
        (Some(&s), Some(&d)) => out.push((
          s,
          d,
          Expr::FunctionCall {
            name: "DirectedEdge".to_string(),
            args: vec![from.clone(), to.clone()].into(),
          },
        )),
        _ => return not_a_graph(),
      }
    }
  }
  out.sort_by_key(|(s, d, _)| (*s, *d));

  let mut new_args = vec![
    Expr::List(vertices),
    Expr::List(out.into_iter().map(|(_, _, e)| e).collect()),
  ];
  new_args.extend(extra);
  Ok(Expr::FunctionCall {
    name: "Graph".to_string(),
    args: new_args.into(),
  })
}

/// FindIndependentVertexSet[g] - one maximum independent vertex set,
/// wrapped in a list. Among all maximum sets wolframscript returns the
/// lexicographically first by vertex-list position (so Graph[{2 <-> 1}]
/// gives {{2}}); directed edges count via the underlying undirected
/// graph. Graphs beyond 64 vertices stay unevaluated.
pub fn find_independent_vertex_set_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "FindIndependentVertexSet".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 1 {
    return Ok(unevaluated(args));
  }
  let graph = match &args[0] {
    Expr::List(_) => {
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Graph".to_string(),
        args: vec![args[0].clone()].into(),
      })?
    }
    other => other.clone(),
  };
  let (vertices, edges) = match &graph {
    Expr::FunctionCall { name, args: gargs }
      if name == "Graph" && gargs.len() >= 2 =>
    {
      match (&gargs[0], &gargs[1]) {
        (Expr::List(v), Expr::List(e)) => (v.clone(), e.clone()),
        _ => return Ok(unevaluated(args)),
      }
    }
    _ => return Ok(unevaluated(args)),
  };
  let n = vertices.len();
  if n == 0 || n > 64 {
    return Ok(unevaluated(args));
  }
  let key = |e: &Expr| crate::syntax::expr_to_string(e);
  let index: std::collections::HashMap<String, usize> = vertices
    .iter()
    .enumerate()
    .map(|(i, v)| (key(v), i))
    .collect();
  // Neighbor bitmasks over the underlying undirected graph
  let mut nbr = vec![0u64; n];
  for edge in edges.iter() {
    if let Expr::FunctionCall { name, args: eargs } = edge
      && (name == "DirectedEdge" || name == "UndirectedEdge")
      && eargs.len() == 2
      && let (Some(&u), Some(&v)) =
        (index.get(&key(&eargs[0])), index.get(&key(&eargs[1])))
    {
      if u != v {
        nbr[u] |= 1 << v;
        nbr[v] |= 1 << u;
      }
    } else {
      return Ok(unevaluated(args));
    }
  }

  // Maximum-independent-set size on the allowed vertex mask
  fn mis(allowed: u64, nbr: &[u64]) -> u32 {
    if allowed == 0 {
      return 0;
    }
    let v = allowed.trailing_zeros() as usize;
    let without = mis(allowed & !(1u64 << v), nbr);
    let with = 1 + mis(allowed & !(1u64 << v) & !nbr[v], nbr);
    with.max(without)
  }

  let full = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
  let target = mis(full, &nbr);
  // Greedy lexicographic selection: take vertex i whenever doing so
  // still extends to a maximum independent set
  let mut chosen: Vec<Expr> = Vec::new();
  let mut taken = 0u32;
  let mut allowed = full;
  for i in 0..n {
    let bit = 1u64 << i;
    if allowed & bit == 0 {
      continue;
    }
    let rest = allowed & !bit & !nbr[i];
    if taken + 1 + mis(rest & !((bit << 1).wrapping_sub(1)), &nbr) == target {
      chosen.push(vertices[i].clone());
      taken += 1;
      allowed = rest;
    }
  }
  Ok(Expr::List(vec![Expr::List(chosen.into())].into()))
}

/// VertexComponent[g, v] / VertexComponent[g, {v1, ...}] - vertices
/// from which the given vertices can be reached (the connected
/// component for undirected graphs), in BFS order seed-first with
/// in-neighbors visited in vertex-list order; multiple seeds expand
/// sequentially with shared visited state. Unknown vertices emit
/// VertexComponent::inv.
pub fn vertex_component_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "VertexComponent".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 2 {
    return Ok(unevaluated(args));
  }
  let graph = match &args[0] {
    Expr::List(_) => {
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Graph".to_string(),
        args: vec![args[0].clone()].into(),
      })?
    }
    other => other.clone(),
  };
  let (vertices, edges) = match &graph {
    Expr::FunctionCall { name, args: gargs }
      if name == "Graph" && gargs.len() >= 2 =>
    {
      match (&gargs[0], &gargs[1]) {
        (Expr::List(v), Expr::List(e)) => (v.clone(), e.clone()),
        _ => return Ok(unevaluated(args)),
      }
    }
    _ => return Ok(unevaluated(args)),
  };
  let key = |e: &Expr| crate::syntax::expr_to_string(e);
  let index: std::collections::HashMap<String, usize> = vertices
    .iter()
    .enumerate()
    .map(|(i, v)| (key(v), i))
    .collect();
  let n = vertices.len();
  // In-neighbor lists (undirected edges count both ways)
  let mut in_nbrs: Vec<Vec<usize>> = vec![Vec::new(); n];
  for edge in edges.iter() {
    if let Expr::FunctionCall { name, args: eargs } = edge
      && (name == "DirectedEdge" || name == "UndirectedEdge")
      && eargs.len() == 2
      && let (Some(&u), Some(&v)) =
        (index.get(&key(&eargs[0])), index.get(&key(&eargs[1])))
    {
      in_nbrs[v].push(u);
      if name == "UndirectedEdge" {
        in_nbrs[u].push(v);
      }
    } else {
      return Ok(unevaluated(args));
    }
  }
  // Visit in-neighbors in vertex-list order
  for list in &mut in_nbrs {
    list.sort_unstable();
    list.dedup();
  }

  let seeds: Vec<&Expr> = match &args[1] {
    Expr::List(items) => items.iter().collect(),
    single => vec![single],
  };
  let mut seed_indices = Vec::with_capacity(seeds.len());
  for seed in &seeds {
    match index.get(&key(seed)) {
      Some(&i) => seed_indices.push(i),
      None => {
        crate::emit_message(&format!(
          "VertexComponent::inv: The argument {} in {} is not a valid vertex.",
          key(seed),
          crate::syntax::expr_to_string(&Expr::FunctionCall {
            name: "VertexComponent".to_string(),
            args: vec![graph.clone(), args[1].clone()].into(),
          })
        ));
        return Ok(unevaluated(&[graph, args[1].clone()]));
      }
    }
  }

  let mut visited = vec![false; n];
  let mut order: Vec<usize> = Vec::new();
  for &start in &seed_indices {
    if visited[start] {
      continue;
    }
    visited[start] = true;
    let mut queue = std::collections::VecDeque::from([start]);
    while let Some(v) = queue.pop_front() {
      order.push(v);
      for &u in &in_nbrs[v] {
        if !visited[u] {
          visited[u] = true;
          queue.push_back(u);
        }
      }
    }
  }
  Ok(Expr::List(
    order.into_iter().map(|i| vertices[i].clone()).collect(),
  ))
}

/// Shared engine for VertexInComponent / VertexOutComponent.
/// `out == true` follows directed edges forwards (vertices reachable *from*
/// the seeds); `out == false` follows them backwards (vertices that can
/// *reach* the seeds). Undirected edges are bidirectional either way. An
/// optional integer third argument bounds the path length. Vertices come out
/// in seed-first BFS order with neighbours visited in vertex-list order.
pub fn vertex_reach_component_ast(
  name: &str,
  args: &[Expr],
  out: bool,
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: name.to_string(),
    args: args.to_vec().into(),
  };
  if args.len() < 2 || args.len() > 3 {
    return Ok(unevaluated());
  }
  // Path-length bound (integer k). The exact-length `{k}` form is not handled.
  let max_depth: usize = match args.get(2) {
    None => usize::MAX,
    Some(Expr::Integer(k)) if *k >= 0 => *k as usize,
    Some(_) => return Ok(unevaluated()),
  };

  let graph = match &args[0] {
    Expr::List(_) => {
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Graph".to_string(),
        args: vec![args[0].clone()].into(),
      })?
    }
    other => other.clone(),
  };
  let (vertices, edges) = match &graph {
    Expr::FunctionCall {
      name: g,
      args: gargs,
    } if g == "Graph" && gargs.len() >= 2 => match (&gargs[0], &gargs[1]) {
      (Expr::List(v), Expr::List(e)) => (v.clone(), e.clone()),
      _ => return Ok(unevaluated()),
    },
    _ => return Ok(unevaluated()),
  };

  let key = |e: &Expr| crate::syntax::expr_to_string(e);
  let index: std::collections::HashMap<String, usize> = vertices
    .iter()
    .enumerate()
    .map(|(i, v)| (key(v), i))
    .collect();
  let n = vertices.len();
  let mut nbrs: Vec<Vec<usize>> = vec![Vec::new(); n];
  for edge in edges.iter() {
    if let Expr::FunctionCall {
      name: en,
      args: eargs,
    } = edge
      && (en == "DirectedEdge" || en == "UndirectedEdge")
      && eargs.len() == 2
      && let (Some(&u), Some(&v)) =
        (index.get(&key(&eargs[0])), index.get(&key(&eargs[1])))
    {
      // Forward search follows u -> v; backward search follows v -> u.
      let (from, to) = if out { (u, v) } else { (v, u) };
      nbrs[from].push(to);
      if en == "UndirectedEdge" {
        nbrs[to].push(from);
      }
    } else {
      return Ok(unevaluated());
    }
  }
  for list in &mut nbrs {
    list.sort_unstable();
    list.dedup();
  }

  let seeds: Vec<&Expr> = match &args[1] {
    Expr::List(items) => items.iter().collect(),
    single => vec![single],
  };
  let mut seed_indices = Vec::with_capacity(seeds.len());
  for seed in &seeds {
    match index.get(&key(seed)) {
      Some(&i) => seed_indices.push(i),
      None => {
        crate::emit_message(&format!(
          "{}::inv: The argument {} in {} is not a valid vertex.",
          name,
          key(seed),
          crate::syntax::expr_to_string(&Expr::FunctionCall {
            name: name.to_string(),
            args: vec![graph.clone(), args[1].clone()].into(),
          })
        ));
        return Ok(unevaluated());
      }
    }
  }

  let mut visited = vec![false; n];
  let mut order: Vec<usize> = Vec::new();
  for &start in &seed_indices {
    if visited[start] {
      continue;
    }
    visited[start] = true;
    let mut queue = std::collections::VecDeque::from([(start, 0usize)]);
    while let Some((v, depth)) = queue.pop_front() {
      order.push(v);
      if depth < max_depth {
        for &u in &nbrs[v] {
          if !visited[u] {
            visited[u] = true;
            queue.push_back((u, depth + 1));
          }
        }
      }
    }
  }
  Ok(Expr::List(
    order.into_iter().map(|i| vertices[i].clone()).collect(),
  ))
}

/// WeightedAdjacencyGraph[wmat] / WeightedAdjacencyGraph[{v...}, wmat]
/// - graph from a weight matrix, with Infinity marking absent edges
/// (zero is a real weight). Symmetric matrices give undirected graphs
/// (upper triangle incl. self-loops, row-major); anything else gives
/// directed edges in row-major order. Weights land in the Graph's
/// EdgeWeight option.
pub fn weighted_adjacency_graph_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "WeightedAdjacencyGraph".to_string(),
    args: args.to_vec().into(),
  };
  let (vertices, matrix) = match args {
    [Expr::List(m)] => (None, m),
    [Expr::List(v), Expr::List(m)] => (Some(v.clone()), m),
    _ => return Ok(unevaluated(args)),
  };
  let n = matrix.len();
  let rows: Option<Vec<&[Expr]>> = matrix
    .iter()
    .map(|row| match row {
      Expr::List(cells) if cells.len() == n => Some(cells.as_slice()),
      _ => None,
    })
    .collect();
  let Some(rows) = rows else {
    return Ok(unevaluated(args));
  };
  let vertices: Vec<Expr> = match vertices {
    Some(v) if v.len() == n => v.to_vec(),
    Some(_) => return Ok(unevaluated(args)),
    None => (1..=n as i128).map(Expr::Integer).collect(),
  };
  let is_edge = |e: &Expr| !matches!(e, Expr::Identifier(s) if s == "Infinity");
  let key = |e: &Expr| crate::syntax::expr_to_string(e);
  let symmetric =
    (0..n).all(|i| (0..n).all(|j| key(&rows[i][j]) == key(&rows[j][i])));

  let mut edges: Vec<Expr> = Vec::new();
  let mut weights: Vec<Expr> = Vec::new();
  for i in 0..n {
    let j_start = if symmetric { i } else { 0 };
    for j in j_start..n {
      if !symmetric && i == j {
        continue;
      }
      if symmetric && i == j && !is_edge(&rows[i][j]) {
        continue;
      }
      if is_edge(&rows[i][j]) {
        edges.push(Expr::FunctionCall {
          name: if symmetric {
            "UndirectedEdge".to_string()
          } else {
            "DirectedEdge".to_string()
          },
          args: vec![vertices[i].clone(), vertices[j].clone()].into(),
        });
        weights.push(rows[i][j].clone());
      }
    }
  }
  Ok(Expr::FunctionCall {
    name: "Graph".to_string(),
    args: vec![
      Expr::List(vertices.into()),
      Expr::List(edges.into()),
      Expr::Rule {
        pattern: Box::new(Expr::Identifier("EdgeWeight".to_string())),
        replacement: Box::new(Expr::List(weights.into())),
      },
    ]
    .into(),
  })
}

/// FindMinimumCostFlow[cmat, s, t] - minimum total cost of a maximum
/// flow from s to t where each nonzero cost-matrix entry is an arc of
/// capacity one (wolframscript's default). No path at all stays
/// unevaluated, matching wolframscript.
pub fn find_minimum_cost_flow_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "FindMinimumCostFlow".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 3 {
    return Ok(unevaluated(args));
  }
  let matrix = match &args[0] {
    Expr::List(rows) if !rows.is_empty() => rows,
    _ => return Ok(unevaluated(args)),
  };
  let n = matrix.len();
  let mut all_integer = true;
  let mut cost = vec![vec![0.0f64; n]; n];
  let mut int_cost = vec![vec![0i128; n]; n];
  for (i, row) in matrix.iter().enumerate() {
    let Expr::List(cells) = row else {
      return Ok(unevaluated(args));
    };
    if cells.len() != n {
      return Ok(unevaluated(args));
    }
    for (j, cell) in cells.iter().enumerate() {
      match cell {
        Expr::Integer(v) => {
          cost[i][j] = *v as f64;
          int_cost[i][j] = *v;
        }
        _ => match crate::functions::try_eval_to_f64(cell) {
          Some(v) => {
            cost[i][j] = v;
            all_integer = false;
          }
          None => return Ok(unevaluated(args)),
        },
      }
    }
  }
  let (s, t) = match (&args[1], &args[2]) {
    (Expr::Integer(a), Expr::Integer(b))
      if *a >= 1 && *a as usize <= n && *b >= 1 && *b as usize <= n =>
    {
      ((*a - 1) as usize, (*b - 1) as usize)
    }
    _ => return Ok(unevaluated(args)),
  };

  // Residual network: arcs with capacity 1 and cost c, reverse arcs
  // with capacity 0 and cost -c
  struct Arc {
    to: usize,
    cap: i32,
    cost: f64,
    rev: usize,
  }
  let mut adj: Vec<Vec<Arc>> = (0..n).map(|_| Vec::new()).collect();
  for i in 0..n {
    for j in 0..n {
      if i != j && cost[i][j] != 0.0 {
        let fwd_rev = adj[j].len();
        let bwd_rev = adj[i].len();
        adj[i].push(Arc {
          to: j,
          cap: 1,
          cost: cost[i][j],
          rev: fwd_rev,
        });
        adj[j].push(Arc {
          to: i,
          cap: 0,
          cost: -cost[i][j],
          rev: bwd_rev,
        });
      }
    }
  }

  // Successive shortest augmenting paths (Bellman-Ford handles the
  // negative residual costs)
  let mut total_int = 0i128;
  let mut augmented_any = false;
  loop {
    let mut dist = vec![f64::INFINITY; n];
    let mut prev: Vec<Option<(usize, usize)>> = vec![None; n];
    dist[s] = 0.0;
    for _ in 0..n {
      let mut changed = false;
      for u in 0..n {
        if dist[u].is_infinite() {
          continue;
        }
        for (k, arc) in adj[u].iter().enumerate() {
          if arc.cap > 0 && dist[u] + arc.cost < dist[arc.to] - 1e-12 {
            dist[arc.to] = dist[u] + arc.cost;
            prev[arc.to] = Some((u, k));
            changed = true;
          }
        }
      }
      if !changed {
        break;
      }
    }
    if dist[t].is_infinite() {
      break;
    }
    augmented_any = true;
    // Augment one unit along the path
    let mut v = t;
    while v != s {
      let (u, k) = prev[v].expect("path edge");
      let arc_rev = adj[u][k].rev;
      adj[u][k].cap -= 1;
      adj[v][arc_rev].cap += 1;
      // Sum the integer costs of forward arcs (reverse arcs subtract)
      if all_integer {
        if int_cost[u][v] != 0 {
          total_int += int_cost[u][v];
        } else {
          total_int -= int_cost[v][u];
        }
      }
      v = u;
    }
  }
  if !augmented_any {
    return Ok(unevaluated(args));
  }
  if all_integer {
    Ok(Expr::Integer(total_int))
  } else {
    // Recompute the real total from saturated forward arcs
    let mut total = 0.0;
    for (u, arcs) in adj.iter().enumerate() {
      for arc in arcs {
        if cost[u][arc.to] != 0.0 && arc.cap == 0 && u != arc.to {
          // forward arc fully used
          total += cost[u][arc.to];
        }
      }
    }
    Ok(Expr::Real(total))
  }
}

/// NearestNeighborGraph[points] / NearestNeighborGraph[points, k] -
/// undirected graph joining every point to its k nearest other points
/// (all equidistant ties included, so {0,1,2,3} links 1 to both
/// neighbors). Vertices keep input order; edges normalize to
/// vertex-order pairs and sort lexicographically. Non-lists emit
/// NearestNeighborGraph::list.
pub fn nearest_neighbor_graph_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "NearestNeighborGraph".to_string(),
    args: args.to_vec().into(),
  };
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated(args));
  }
  let points = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      crate::emit_message(&format!(
        "NearestNeighborGraph::list: List expected at position 1 in {}.",
        crate::syntax::expr_to_string(&unevaluated(args))
      ));
      return Ok(unevaluated(args));
    }
  };
  let k = match args.get(1) {
    None => 1usize,
    Some(Expr::Integer(k)) if *k >= 1 => *k as usize,
    Some(_) => return Ok(unevaluated(args)),
  };

  // Coordinates: scalars or equal-length numeric vectors; exact i128
  // when everything is an integer (ties matter), f64 otherwise
  let coords_of = |p: &Expr| -> Vec<Expr> {
    match p {
      Expr::List(cs) => cs.to_vec(),
      other => vec![other.clone()],
    }
  };
  let n = points.len();
  let mut coords: Vec<Vec<Expr>> = Vec::with_capacity(n);
  for p in points.iter() {
    let cs = coords_of(p);
    if !coords.is_empty() && cs.len() != coords[0].len() {
      return Ok(unevaluated(args));
    }
    coords.push(cs);
  }
  let all_integer = coords
    .iter()
    .all(|cs| cs.iter().all(|c| matches!(c, Expr::Integer(_))));

  // Squared distances, exact when possible (scaled by 2^20 for floats
  // only for ordering, never printed)
  let dist2 = |a: &[Expr], b: &[Expr]| -> Option<f64> {
    let mut acc = 0.0;
    for (x, y) in a.iter().zip(b) {
      let xv = crate::functions::try_eval_to_f64(x)?;
      let yv = crate::functions::try_eval_to_f64(y)?;
      acc += (xv - yv) * (xv - yv);
    }
    Some(acc)
  };
  let dist2_int = |a: &[Expr], b: &[Expr]| -> i128 {
    a.iter()
      .zip(b)
      .map(|(x, y)| match (x, y) {
        (Expr::Integer(p), Expr::Integer(q)) => (p - q) * (p - q),
        _ => unreachable!(),
      })
      .sum()
  };

  let mut pair_set: std::collections::BTreeSet<(usize, usize)> =
    std::collections::BTreeSet::new();
  for i in 0..n {
    // distances to all other points
    if all_integer {
      let mut ds: Vec<(i128, usize)> = (0..n)
        .filter(|&j| j != i)
        .map(|j| (dist2_int(&coords[i], &coords[j]), j))
        .collect();
      ds.sort();
      if ds.is_empty() {
        continue;
      }
      let cutoff = ds[(k - 1).min(ds.len() - 1)].0;
      for &(d, j) in &ds {
        if d <= cutoff {
          pair_set.insert((i.min(j), i.max(j)));
        }
      }
    } else {
      let mut ds: Vec<(f64, usize)> = Vec::new();
      for j in 0..n {
        if j == i {
          continue;
        }
        match dist2(&coords[i], &coords[j]) {
          Some(d) => ds.push((d, j)),
          None => return Ok(unevaluated(args)),
        }
      }
      ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
      if ds.is_empty() {
        continue;
      }
      let cutoff = ds[(k - 1).min(ds.len() - 1)].0;
      for &(d, j) in &ds {
        if d <= cutoff {
          pair_set.insert((i.min(j), i.max(j)));
        }
      }
    }
  }
  let edges: Vec<Expr> = pair_set
    .into_iter()
    .map(|(a, b)| Expr::FunctionCall {
      name: "UndirectedEdge".to_string(),
      args: vec![points[a].clone(), points[b].clone()].into(),
    })
    .collect();
  Ok(Expr::FunctionCall {
    name: "Graph".to_string(),
    args: vec![Expr::List(points.clone()), Expr::List(edges.into())].into(),
  })
}

// ---------------------------------------------------------------------------
// EdgeConnectivity / VertexConnectivity via unit-capacity max-flow
// (Edmonds-Karp). Matches wolframscript:
// - EdgeConnectivity[g]: min over t of maxflow(v0, t); single-vertex
//   graphs stay unevaluated (wolframscript artifact)
// - VertexConnectivity[g]: n-1 for complete graphs, else min vertex cut
//   over non-adjacent pairs (vertex-splitting reduction)
// - 3-arg s-t forms; s == t returns VertexDegree[g, s] (edge) or
//   EdgeCount[g] (vertex), adjacent s, t give vertex connectivity 0 —
//   all replicating wolframscript's observed behavior
// - invalid vertices emit `inv` messages naming the argument position

fn bfs_max_flow(cap: &mut [Vec<i64>], s: usize, t: usize) -> i64 {
  let n = cap.len();
  let mut flow = 0;
  loop {
    let mut parent = vec![usize::MAX; n];
    parent[s] = s;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(s);
    while let Some(u) = queue.pop_front() {
      if u == t {
        break;
      }
      for v in 0..n {
        if parent[v] == usize::MAX && cap[u][v] > 0 {
          parent[v] = u;
          queue.push_back(v);
        }
      }
    }
    if parent[t] == usize::MAX {
      return flow;
    }
    let mut bottleneck = i64::MAX;
    let mut v = t;
    while v != s {
      let u = parent[v];
      bottleneck = bottleneck.min(cap[u][v]);
      v = u;
    }
    let mut v = t;
    while v != s {
      let u = parent[v];
      cap[u][v] -= bottleneck;
      cap[v][u] += bottleneck;
      v = u;
    }
    flow += bottleneck;
  }
}

/// Max number of edge-disjoint paths between s and t (unit edge capacities).
fn edge_maxflow(n: usize, pairs: &[(usize, usize)], s: usize, t: usize) -> i64 {
  let mut cap = vec![vec![0i64; n]; n];
  for &(a, b) in pairs {
    if a != b {
      cap[a][b] += 1;
      cap[b][a] += 1;
    }
  }
  bfs_max_flow(&mut cap, s, t)
}

/// Max number of internally vertex-disjoint paths between non-adjacent
/// s and t (vertex-splitting: v_in = v, v_out = v + n, unit vertex caps).
fn vertex_maxflow(
  n: usize,
  pairs: &[(usize, usize)],
  s: usize,
  t: usize,
) -> i64 {
  let inf = (n as i64 + 1) * 2;
  let mut cap = vec![vec![0i64; 2 * n]; 2 * n];
  for v in 0..n {
    cap[v][v + n] = if v == s || v == t { inf } else { 1 };
  }
  for &(a, b) in pairs {
    if a != b {
      cap[a + n][b] = inf;
      cap[b + n][a] = inf;
    }
  }
  bfs_max_flow(&mut cap, s + n, t)
}

/// Directed variant of `vertex_maxflow`: `arcs` are one-way, so arc a -> b only
/// links the out-side of a to the in-side of b.
fn directed_vertex_maxflow(
  n: usize,
  arcs: &[(usize, usize)],
  s: usize,
  t: usize,
) -> i64 {
  let inf = (n as i64 + 1) * 2;
  let mut cap = vec![vec![0i64; 2 * n]; 2 * n];
  for v in 0..n {
    cap[v][v + n] = if v == s || v == t { inf } else { 1 };
  }
  for &(a, b) in arcs {
    if a != b {
      cap[a + n][b] = inf;
    }
  }
  bfs_max_flow(&mut cap, s + n, t)
}

pub fn connectivity_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: name.to_string(),
    args: args.to_vec().into(),
  };
  // Extract Graph[vertex list, undirected edge list]
  let (vertices, edge_exprs) = match &args[0] {
    Expr::FunctionCall {
      name: gname,
      args: gargs,
    } if gname == "Graph" && gargs.len() >= 2 => match (&gargs[0], &gargs[1]) {
      (Expr::List(v), Expr::List(e)) => (v, e),
      _ => return Ok(unevaluated()),
    },
    _ => return Ok(unevaluated()),
  };
  let vkeys: Vec<String> = vertices.iter().map(expr_to_string).collect();
  let n = vkeys.len();
  let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(edge_exprs.len());
  // Directed arcs: a directed edge a -> b is one arc; an undirected edge is
  // two. Only used when the graph actually contains a directed edge.
  let mut arcs: Vec<(usize, usize)> = Vec::new();
  let mut directed = false;
  for e in edge_exprs.iter() {
    match e {
      Expr::FunctionCall {
        name: ename,
        args: eargs,
      } if (ename == "UndirectedEdge" || ename == "DirectedEdge")
        && eargs.len() == 2 =>
      {
        let a = vkeys.iter().position(|k| *k == expr_to_string(&eargs[0]));
        let b = vkeys.iter().position(|k| *k == expr_to_string(&eargs[1]));
        match (a, b) {
          (Some(a), Some(b)) => {
            pairs.push((a, b));
            if ename == "DirectedEdge" {
              directed = true;
              arcs.push((a, b));
            } else {
              arcs.push((a, b));
              arcs.push((b, a));
            }
          }
          _ => return Ok(unevaluated()),
        }
      }
      _ => return Ok(unevaluated()),
    }
  }

  // Directed graphs: EdgeConnectivity[g] is the smallest s-t edge cut, and
  // VertexConnectivity[g] the smallest s-t vertex cut, over all ordered pairs
  // (0 when the graph is not strongly connected). Other forms stay unevaluated.
  if directed {
    if name == "EdgeConnectivity" && args.len() == 1 {
      if n <= 1 {
        return Ok(unevaluated());
      }
      let mut min_flow = i64::MAX;
      for s in 0..n {
        for t in 0..n {
          if s != t {
            let mut cap = vec![vec![0i64; n]; n];
            for &(a, b) in &arcs {
              if a != b {
                cap[a][b] += 1;
              }
            }
            min_flow = min_flow.min(bfs_max_flow(&mut cap, s, t));
          }
        }
      }
      return Ok(Expr::Integer(min_flow as i128));
    }
    if name == "VertexConnectivity" && args.len() == 1 {
      if n <= 1 {
        return Ok(Expr::Integer(0));
      }
      let is_arc =
        |s: usize, t: usize| arcs.iter().any(|&(a, b)| a == s && b == t);
      let mut best: Option<i64> = None;
      for s in 0..n {
        for t in 0..n {
          if s != t && !is_arc(s, t) {
            let f = directed_vertex_maxflow(n, &arcs, s, t);
            best = Some(best.map_or(f, |b| b.min(f)));
            if f == 0 {
              return Ok(Expr::Integer(0));
            }
          }
        }
      }
      // Every ordered pair is adjacent: complete digraph, connectivity n - 1.
      return Ok(Expr::Integer(best.unwrap_or(n as i64 - 1) as i128));
    }
    return Ok(unevaluated());
  }

  let adjacent = |s: usize, t: usize| {
    pairs
      .iter()
      .any(|&(a, b)| (a == s && b == t) || (a == t && b == s))
  };

  if args.len() == 3 {
    let find = |e: &Expr| vkeys.iter().position(|k| *k == expr_to_string(e));
    let inv = |pos: usize| {
      crate::emit_message(&format!(
        "{}::inv: The argument {} in {} is not a valid vertex.",
        name,
        pos,
        expr_to_string(&Expr::FunctionCall {
          name: name.to_string(),
          args: args.to_vec().into(),
        })
      ));
    };
    let s = match find(&args[1]) {
      Some(i) => i,
      None => {
        inv(2);
        return Ok(unevaluated());
      }
    };
    let t = match find(&args[2]) {
      Some(i) => i,
      None => {
        inv(3);
        return Ok(unevaluated());
      }
    };
    if s == t {
      // wolframscript artifacts: degree of s (edge), edge count (vertex)
      return Ok(Expr::Integer(if name == "EdgeConnectivity" {
        pairs.iter().filter(|&&(a, b)| a == s || b == s).count() as i128
      } else {
        pairs.len() as i128
      }));
    }
    let result = if name == "EdgeConnectivity" {
      edge_maxflow(n, &pairs, s, t)
    } else if adjacent(s, t) {
      0
    } else {
      vertex_maxflow(n, &pairs, s, t)
    };
    return Ok(Expr::Integer(result as i128));
  }

  // Single-argument forms
  if name == "EdgeConnectivity" {
    if n <= 1 {
      return Ok(unevaluated());
    }
    let min_flow = (1..n)
      .map(|t| edge_maxflow(n, &pairs, 0, t))
      .min()
      .unwrap_or(0);
    return Ok(Expr::Integer(min_flow as i128));
  }
  // VertexConnectivity
  if n <= 1 {
    return Ok(Expr::Integer(0));
  }
  let mut best: Option<i64> = None;
  for s in 0..n {
    for t in (s + 1)..n {
      if !adjacent(s, t) {
        let f = vertex_maxflow(n, &pairs, s, t);
        best = Some(best.map_or(f, |b| b.min(f)));
        if f == 0 {
          return Ok(Expr::Integer(0));
        }
      }
    }
  }
  // All pairs adjacent: complete graph, connectivity n - 1
  Ok(Expr::Integer(best.unwrap_or(n as i64 - 1) as i128))
}

// ---------------------------------------------------------------------------
// KCoreComponents[g, k] — connected components of the k-core (the maximal
// subgraph in which every vertex has degree >= k), each as a vertex list.
// Matches wolframscript:
// - components are discovered in reverse VertexList order, members listed
//   in VertexList order, then stable-sorted by size descending
// - third argument must be "In" or "Out" (degree direction; identical to
//   the plain form on undirected graphs), anything else emits ::inv
// - non-integer k emits ::int; non-graph first arguments stay unevaluated
// Note: wolframscript's k <= 0 multi-component ordering is an internal
// artifact that is not replicated; k <= 0 performs no pruning here.

pub fn k_core_components_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "KCoreComponents".to_string(),
    args: args.to_vec().into(),
  };
  let call_str = || {
    crate::syntax::format_expr(
      &Expr::FunctionCall {
        name: "KCoreComponents".to_string(),
        args: args.to_vec().into(),
      },
      crate::syntax::ExprForm::Output,
    )
  };
  let (vertices, edge_exprs) = match &args[0] {
    Expr::FunctionCall {
      name: gname,
      args: gargs,
    } if gname == "Graph" && gargs.len() >= 2 => match (&gargs[0], &gargs[1]) {
      (Expr::List(v), Expr::List(e)) => (v, e),
      _ => return Ok(unevaluated()),
    },
    _ => return Ok(unevaluated()),
  };
  let k = match &args[1] {
    Expr::Integer(k) => *k,
    _ => {
      crate::emit_message(&format!(
        "KCoreComponents::int: Integer expected at position 2 in {}.",
        call_str()
      ));
      return Ok(unevaluated());
    }
  };
  if args.len() == 3 {
    let valid = matches!(&args[2], Expr::String(s) if s == "In" || s == "Out");
    if !valid {
      crate::emit_message(&format!(
        "KCoreComponents::inv: The argument {} in {} is not a valid parameter.",
        crate::syntax::format_expr(&args[2], crate::syntax::ExprForm::Output),
        call_str()
      ));
      return Ok(unevaluated());
    }
  }

  let vkeys: Vec<String> = vertices.iter().map(expr_to_string).collect();
  let n = vkeys.len();
  // Underlying simple graph: dedup edges, drop self-loops
  let mut adj: Vec<std::collections::BTreeSet<usize>> =
    vec![Default::default(); n];
  for e in edge_exprs.iter() {
    if let Expr::FunctionCall {
      name: ename,
      args: eargs,
    } = e
      && ename == "UndirectedEdge"
      && eargs.len() == 2
    {
      let a = vkeys.iter().position(|v| *v == expr_to_string(&eargs[0]));
      let b = vkeys.iter().position(|v| *v == expr_to_string(&eargs[1]));
      if let (Some(a), Some(b)) = (a, b) {
        if a != b {
          adj[a].insert(b);
          adj[b].insert(a);
        }
      } else {
        return Ok(unevaluated());
      }
    } else {
      return Ok(unevaluated());
    }
  }

  // Prune to the k-core: repeatedly remove vertices with degree < k
  let mut alive = vec![true; n];
  if k > 0 {
    loop {
      let mut changed = false;
      for v in 0..n {
        if alive[v]
          && (adj[v].iter().filter(|&&u| alive[u]).count() as i128) < k
        {
          alive[v] = false;
          changed = true;
        }
      }
      if !changed {
        break;
      }
    }
  }

  // Connected components; members listed in VertexList order
  let mut seen = vec![false; n];
  let mut components: Vec<Vec<usize>> = Vec::new();
  for start in 0..n {
    if !alive[start] || seen[start] {
      continue;
    }
    let mut members = Vec::new();
    let mut queue = std::collections::VecDeque::new();
    seen[start] = true;
    queue.push_back(start);
    while let Some(u) = queue.pop_front() {
      members.push(u);
      for &v in &adj[u] {
        if alive[v] && !seen[v] {
          seen[v] = true;
          queue.push_back(v);
        }
      }
    }
    members.sort_unstable();
    components.push(members);
  }
  // wolframscript order: size descending, ties broken by the position of
  // the first member in the vertex list, descending
  components
    .sort_by_key(|c| (std::cmp::Reverse(c.len()), std::cmp::Reverse(c[0])));

  Ok(Expr::List(
    components
      .into_iter()
      .map(|c| Expr::List(c.into_iter().map(|i| vertices[i].clone()).collect()))
      .collect::<Vec<_>>()
      .into(),
  ))
}

// ---------------------------------------------------------------------------
// FindClique[g], FindClique[g, spec], FindClique[g, spec, count] — maximal
// cliques (cliques not contained in any larger clique), matching
// wolframscript's conventions (decoded by probing):
// - spec: n (size <= n), {n} (exactly n), {min, max}, Infinity; sizes are
//   filtered AFTER maximality is determined in the whole graph
// - count 1 (default): the largest qualifying clique, ties broken by
//   ascending lexicographic vertex order
// - count k >= 2: the first k maximal cliques in ascending lexicographic
//   enumeration order, then sorted by size descending / lex descending
// - count All: all qualifying cliques, size descending / lex descending
// - isolated vertices are maximal 1-cliques; invalid specs emit ::inv

fn bron_kerbosch(
  adj: &[Vec<bool>],
  r: &mut Vec<usize>,
  p: &mut Vec<usize>,
  x: &mut Vec<usize>,
  out: &mut Vec<Vec<usize>>,
) {
  if p.is_empty() && x.is_empty() {
    out.push(r.clone());
    return;
  }
  // Pivot: vertex in P ∪ X with most neighbors in P
  let pivot = p
    .iter()
    .chain(x.iter())
    .copied()
    .max_by_key(|&u| p.iter().filter(|&&v| adj[u][v]).count());
  let candidates: Vec<usize> = match pivot {
    Some(u) => p.iter().copied().filter(|&v| !adj[u][v]).collect(),
    None => p.clone(),
  };
  for v in candidates {
    r.push(v);
    let mut new_p: Vec<usize> =
      p.iter().copied().filter(|&w| adj[v][w]).collect();
    let mut new_x: Vec<usize> =
      x.iter().copied().filter(|&w| adj[v][w]).collect();
    bron_kerbosch(adj, r, &mut new_p, &mut new_x, out);
    r.pop();
    p.retain(|&w| w != v);
    x.push(v);
  }
}

pub fn find_clique_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "FindClique".to_string(),
    args: args.to_vec().into(),
  };
  let call_str = || {
    crate::syntax::format_expr(
      &Expr::FunctionCall {
        name: "FindClique".to_string(),
        args: args.to_vec().into(),
      },
      crate::syntax::ExprForm::Output,
    )
  };
  let inv = |arg: &Expr| {
    crate::emit_message(&format!(
      "FindClique::inv: The argument {} in {} is not a valid parameter.",
      crate::syntax::format_expr(arg, crate::syntax::ExprForm::Output),
      call_str()
    ));
  };
  let (vertices, edge_exprs) = match &args[0] {
    Expr::FunctionCall {
      name: gname,
      args: gargs,
    } if gname == "Graph" && gargs.len() >= 2 => match (&gargs[0], &gargs[1]) {
      (Expr::List(v), Expr::List(e)) => (v, e),
      _ => return Ok(unevaluated()),
    },
    _ => return Ok(unevaluated()),
  };

  // Size specification
  let is_infinity = |e: &Expr| {
    matches!(e, Expr::Identifier(s) if s == "Infinity")
      || matches!(e, Expr::FunctionCall { name, .. } if name == "DirectedInfinity")
  };
  let (min_size, max_size): (usize, usize) = if args.len() >= 2 {
    match &args[1] {
      Expr::Integer(n) if *n >= 1 => (1, *n as usize),
      e if is_infinity(e) => (1, usize::MAX),
      Expr::List(items) if items.len() == 1 => match &items[0] {
        Expr::Integer(n) if *n >= 1 => (*n as usize, *n as usize),
        _ => {
          inv(&args[1]);
          return Ok(unevaluated());
        }
      },
      Expr::List(items) if items.len() == 2 => match (&items[0], &items[1]) {
        (Expr::Integer(lo), Expr::Integer(hi)) if *lo >= 1 && *hi >= *lo => {
          (*lo as usize, *hi as usize)
        }
        (Expr::Integer(lo), e) if *lo >= 1 && is_infinity(e) => {
          (*lo as usize, usize::MAX)
        }
        _ => {
          inv(&args[1]);
          return Ok(unevaluated());
        }
      },
      other => {
        inv(other);
        return Ok(unevaluated());
      }
    }
  } else {
    (1, usize::MAX)
  };

  // Count specification
  enum Count {
    One,
    K(usize),
    All,
  }
  let count = if args.len() == 3 {
    match &args[2] {
      Expr::Integer(1) => Count::One,
      Expr::Integer(k) if *k >= 2 => Count::K(*k as usize),
      Expr::Identifier(s) if s == "All" => Count::All,
      e if is_infinity(e) => Count::All,
      other => {
        inv(other);
        return Ok(unevaluated());
      }
    }
  } else {
    Count::One
  };

  let n = vertices.len();
  let mut adj = vec![vec![false; n]; n];
  for e in edge_exprs.iter() {
    if let Expr::FunctionCall {
      name: ename,
      args: eargs,
    } = e
      && ename == "UndirectedEdge"
      && eargs.len() == 2
    {
      let vkey = |x: &Expr| {
        vertices
          .iter()
          .position(|v| expr_to_string(v) == expr_to_string(x))
      };
      match (vkey(&eargs[0]), vkey(&eargs[1])) {
        (Some(a), Some(b)) if a != b => {
          adj[a][b] = true;
          adj[b][a] = true;
        }
        (Some(_), Some(_)) => {}
        _ => return Ok(unevaluated()),
      }
    } else {
      return Ok(unevaluated());
    }
  }

  // All maximal cliques (members sorted ascending), then filter by size
  let mut cliques: Vec<Vec<usize>> = Vec::new();
  let mut p: Vec<usize> = (0..n).collect();
  bron_kerbosch(&adj, &mut Vec::new(), &mut p, &mut Vec::new(), &mut cliques);
  for c in cliques.iter_mut() {
    c.sort_unstable();
  }
  let mut qualifying: Vec<Vec<usize>> = cliques
    .into_iter()
    .filter(|c| c.len() >= min_size && c.len() <= max_size)
    .collect();
  // Ascending lexicographic enumeration order
  qualifying.sort();

  let selected: Vec<Vec<usize>> = match count {
    Count::One => {
      let best_len = qualifying.iter().map(|c| c.len()).max();
      match best_len {
        Some(l) => {
          vec![
            qualifying
              .iter()
              .find(|c| c.len() == l)
              .cloned()
              .unwrap_or_default(),
          ]
        }
        None => Vec::new(),
      }
    }
    Count::K(k) => {
      let mut taken: Vec<Vec<usize>> = qualifying.into_iter().take(k).collect();
      taken.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| b.cmp(a)));
      taken
    }
    Count::All => {
      qualifying.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| b.cmp(a)));
      qualifying
    }
  };

  Ok(Expr::List(
    selected
      .into_iter()
      .map(|c| {
        Expr::List(
          c.into_iter()
            .map(|i| vertices[i].clone())
            .collect::<Vec<_>>()
            .into(),
        )
      })
      .collect::<Vec<_>>()
      .into(),
  ))
}

// ---------------------------------------------------------------------------
// Subgraph[g, vertices] — the induced subgraph. The vertex list keeps the
// given order (unknown vertices are ignored); edges are sorted by the
// (max, min) positions of their endpoints in the given list, with the
// earlier-position endpoint first — matching wolframscript for generator
// graphs (CycleGraph, PetersenGraph, ...). Note: wolframscript's edge
// display order for explicitly-constructed graphs is an igraph artifact
// that is not replicated.
pub fn subgraph_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "Subgraph".to_string(),
    args: args.to_vec().into(),
  };
  let (vertices, edge_exprs) = match &args[0] {
    Expr::FunctionCall {
      name: gname,
      args: gargs,
    } if gname == "Graph" && gargs.len() >= 2 => match (&gargs[0], &gargs[1]) {
      (Expr::List(v), Expr::List(e)) => (v, e),
      _ => return Ok(unevaluated()),
    },
    _ => return Ok(unevaluated()),
  };
  // The vertex spec: a list, or a single vertex
  let requested: Vec<Expr> = match &args[1] {
    Expr::List(items) => items.iter().cloned().collect(),
    single => vec![single.clone()],
  };
  // Keep only vertices of g, in the requested order, deduplicated
  let mut sub_vertices: Vec<Expr> = Vec::new();
  let mut sub_keys: Vec<String> = Vec::new();
  for r in requested {
    let key = expr_to_string(&r);
    if vertices.iter().any(|v| expr_to_string(v) == key)
      && !sub_keys.contains(&key)
    {
      sub_keys.push(key);
      sub_vertices.push(r);
    }
  }
  let pos_of = |e: &Expr| sub_keys.iter().position(|k| *k == expr_to_string(e));
  let mut edges: Vec<((usize, usize), Expr)> = Vec::new();
  for e in edge_exprs.iter() {
    if let Expr::FunctionCall {
      name: ename,
      args: eargs,
    } = e
      && ename == "UndirectedEdge"
      && eargs.len() == 2
      && let (Some(pa), Some(pb)) = (pos_of(&eargs[0]), pos_of(&eargs[1]))
    {
      let (first, second) = if pa <= pb {
        (eargs[0].clone(), eargs[1].clone())
      } else {
        (eargs[1].clone(), eargs[0].clone())
      };
      edges.push((
        (pa.max(pb), pa.min(pb)),
        Expr::FunctionCall {
          name: "UndirectedEdge".to_string(),
          args: vec![first, second].into(),
        },
      ));
    }
  }
  edges.sort_by_key(|(k, _)| *k);
  Ok(Expr::FunctionCall {
    name: "Graph".to_string(),
    args: vec![
      Expr::List(sub_vertices.into()),
      Expr::List(edges.into_iter().map(|(_, e)| e).collect::<Vec<_>>().into()),
    ]
    .into(),
  })
}

/// KirchhoffGraph[m] / KirchhoffGraph[vertices, m] — build the graph whose
/// Kirchhoff (Laplacian) matrix is m. Off-diagonal entries must be
/// non-positive integers; -k means k parallel edges (the diagonal is not
/// validated, matching wolframscript). A symmetric matrix gives undirected
/// edges from the upper triangle; a non-symmetric one gives directed edges
/// row by row. Trailing arguments (Graph options) pass through.
pub fn kirchhoff_graph_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "KirchhoffGraph".to_string(),
    args: args.to_vec().into(),
  };
  if args.is_empty() {
    return Ok(unevaluated());
  }
  // Optional leading vertex-name list, then the matrix, then options.
  let (names, matrix_expr, matrix_pos, extra) = match args {
    [Expr::List(v), m @ Expr::List(rows), rest @ ..]
      if v.iter().all(|e| !matches!(e, Expr::List(_)))
        && rows.iter().all(|r| matches!(r, Expr::List(_))) =>
    {
      (Some(v), m, 2, rest)
    }
    [m, rest @ ..] => (None, m, 1, rest),
    _ => return Ok(unevaluated()),
  };

  let not_square = || {
    crate::emit_message(&format!(
      "KirchhoffGraph::matsq: Argument {} at position {} is not a nonempty square matrix.",
      crate::syntax::expr_to_output(matrix_expr),
      matrix_pos
    ));
    Ok(unevaluated())
  };
  let invalid = || {
    crate::emit_message(&format!(
      "KirchhoffGraph::inv: The argument {} in {} is not a valid Kirchhoff matrix.",
      crate::syntax::expr_to_output(matrix_expr),
      crate::syntax::expr_to_string(&Expr::FunctionCall {
        name: "KirchhoffGraph".to_string(),
        args: vec![matrix_expr.clone()].into(),
      })
    ));
    Ok(unevaluated())
  };
  let Expr::List(rows) = matrix_expr else {
    return not_square();
  };
  let n = rows.len();
  let mut matrix: Vec<Vec<i64>> = Vec::with_capacity(n);
  for row in rows.iter() {
    let Expr::List(cells) = row else {
      return not_square();
    };
    if cells.len() != n {
      return not_square();
    }
    let mut out = Vec::with_capacity(n);
    for cell in cells.iter() {
      match cell {
        Expr::Integer(v) => match i64::try_from(*v) {
          Ok(v) => out.push(v),
          Err(_) => return invalid(),
        },
        _ => return invalid(),
      }
    }
    matrix.push(out);
  }
  if n == 0 {
    return not_square();
  }
  for (i, row) in matrix.iter().enumerate() {
    for (j, &v) in row.iter().enumerate() {
      if i != j && v > 0 {
        return invalid();
      }
    }
  }

  let vertices: Vec<Expr> = match names {
    Some(v) => {
      if v.len() != n {
        return Ok(unevaluated());
      }
      v.to_vec()
    }
    None => (1..=n as i128).map(Expr::Integer).collect(),
  };
  let symmetric = (0..n).all(|i| (0..n).all(|j| matrix[i][j] == matrix[j][i]));
  let mut edges: Vec<Expr> = Vec::new();
  for i in 0..n {
    for j in 0..n {
      if i == j || matrix[i][j] >= 0 || (symmetric && j < i) {
        continue;
      }
      let head = if symmetric {
        "UndirectedEdge"
      } else {
        "DirectedEdge"
      };
      for _ in 0..(-matrix[i][j]) {
        edges.push(Expr::FunctionCall {
          name: head.to_string(),
          args: vec![vertices[i].clone(), vertices[j].clone()].into(),
        });
      }
    }
  }
  let mut graph_args =
    vec![Expr::List(vertices.into()), Expr::List(edges.into())];
  graph_args.extend(extra.iter().cloned());
  Ok(Expr::FunctionCall {
    name: "Graph".to_string(),
    args: graph_args.into(),
  })
}

/// IncidenceGraph[m] / IncidenceGraph[vertices, m] — build the graph whose
/// vertex-by-edge incidence matrix is m. Recognized column patterns are the
/// ones IncidenceMatrix itself produces: two 1s → undirected edge, a -1/+1
/// pair → directed edge, a single 2 → undirected self-loop, a single -2 →
/// directed self-loop. wolframscript emits directed edges first (in column
/// order, -2 loops in place), then undirected non-loop edges, then the
/// undirected self-loops.
pub fn incidence_graph_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "IncidenceGraph".to_string(),
    args: args.to_vec().into(),
  };
  let (names, matrix_expr, matrix_pos) = match args {
    [m] => (None, m, 1),
    [Expr::List(v), m] => (Some(v), m, 2),
    _ => return Ok(unevaluated()),
  };

  // The matrix must be a nonempty rectangular list of integer lists.
  let not_a_matrix = || {
    crate::emit_message(&format!(
      "IncidenceGraph::matrix: Argument {} at position {} is not a nonempty rectangular matrix.",
      crate::syntax::expr_to_output(matrix_expr),
      matrix_pos
    ));
    Ok(unevaluated())
  };
  let invalid = || {
    crate::emit_message(&format!(
      "IncidenceGraph::inv: The argument {} in {} is not a valid incidence matrix.",
      crate::syntax::expr_to_output(matrix_expr),
      crate::syntax::expr_to_string(&unevaluated())
    ));
    Ok(unevaluated())
  };
  let Expr::List(rows) = matrix_expr else {
    return not_a_matrix();
  };
  let mut matrix: Vec<Vec<&Expr>> = Vec::with_capacity(rows.len());
  for row in rows.iter() {
    match row {
      Expr::List(cells) if !cells.is_empty() => {
        matrix.push(cells.iter().collect())
      }
      _ => return not_a_matrix(),
    }
  }
  if matrix.is_empty() || matrix.iter().any(|r| r.len() != matrix[0].len()) {
    return not_a_matrix();
  }
  let n = matrix.len();
  let n_cols = matrix[0].len();
  let int_matrix: Option<Vec<Vec<i64>>> = matrix
    .iter()
    .map(|r| {
      r.iter()
        .map(|c| match c {
          Expr::Integer(v) => i64::try_from(*v).ok(),
          _ => None,
        })
        .collect()
    })
    .collect();
  let Some(int_matrix) = int_matrix else {
    return invalid();
  };

  // A row-name list of the wrong length stays silently unevaluated.
  let vertices: Vec<Expr> = match names {
    Some(v) => {
      if v.len() != n {
        return Ok(unevaluated());
      }
      v.to_vec()
    }
    None => (1..=n as i128).map(Expr::Integer).collect(),
  };

  let edge = |head: &str, i: usize, j: usize| Expr::FunctionCall {
    name: head.to_string(),
    args: vec![vertices[i].clone(), vertices[j].clone()].into(),
  };
  let mut directed: Vec<Expr> = Vec::new();
  let mut undirected: Vec<Expr> = Vec::new();
  let mut loops: Vec<Expr> = Vec::new();
  for c in 0..n_cols {
    let nonzero: Vec<(usize, i64)> = (0..n)
      .filter(|&r| int_matrix[r][c] != 0)
      .map(|r| (r, int_matrix[r][c]))
      .collect();
    match nonzero.as_slice() {
      [(i, 1), (j, 1)] => undirected.push(edge("UndirectedEdge", *i, *j)),
      [(i, -1), (j, 1)] => directed.push(edge("DirectedEdge", *i, *j)),
      [(i, 1), (j, -1)] => directed.push(edge("DirectedEdge", *j, *i)),
      [(i, 2)] => loops.push(edge("UndirectedEdge", *i, *i)),
      [(i, -2)] => directed.push(edge("DirectedEdge", *i, *i)),
      _ => return invalid(),
    }
  }
  let mut edges = directed;
  edges.append(&mut undirected);
  edges.append(&mut loops);
  Ok(Expr::FunctionCall {
    name: "Graph".to_string(),
    args: vec![Expr::List(vertices.into()), Expr::List(edges.into())].into(),
  })
}

// LineGraph[g] — the line graph: one vertex per edge of g (numbered by
// EdgeList position), with two vertices adjacent when the underlying
// edges share an endpoint. Edges are listed in ascending canonical order,
// matching wolframscript for generator graphs (the reversed display
// orientation wolframscript uses for explicitly-constructed graphs is an
// igraph artifact that is not replicated).
pub fn line_graph_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "LineGraph".to_string(),
    args: args.to_vec().into(),
  };
  let edge_exprs = match &args[0] {
    Expr::FunctionCall {
      name: gname,
      args: gargs,
    } if gname == "Graph" && gargs.len() >= 2 => match &gargs[1] {
      Expr::List(e) => e,
      _ => return Ok(unevaluated()),
    },
    _ => return Ok(unevaluated()),
  };
  let mut endpoints: Vec<(String, String)> = Vec::new();
  for e in edge_exprs.iter() {
    if let Expr::FunctionCall {
      name: ename,
      args: eargs,
    } = e
      && ename == "UndirectedEdge"
      && eargs.len() == 2
    {
      endpoints.push((expr_to_string(&eargs[0]), expr_to_string(&eargs[1])));
    } else {
      return Ok(unevaluated());
    }
  }
  let m = endpoints.len();
  let vertices: Vec<Expr> = (1..=m).map(|i| Expr::Integer(i as i128)).collect();
  let mut edges: Vec<Expr> = Vec::new();
  for i in 0..m {
    for j in (i + 1)..m {
      let (a1, b1) = &endpoints[i];
      let (a2, b2) = &endpoints[j];
      if a1 == a2 || a1 == b2 || b1 == a2 || b1 == b2 {
        edges.push(Expr::FunctionCall {
          name: "UndirectedEdge".to_string(),
          args: vec![
            Expr::Integer((i + 1) as i128),
            Expr::Integer((j + 1) as i128),
          ]
          .into(),
        });
      }
    }
  }
  Ok(Expr::FunctionCall {
    name: "Graph".to_string(),
    args: vec![Expr::List(vertices.into()), Expr::List(edges.into())].into(),
  })
}

// ---------------------------------------------------------------------------
// NeighborhoodGraph[g, v], NeighborhoodGraph[g, {v1, ...}, r] — the induced
// subgraph on the vertices within distance r (default 1) of the centers.
// Matches wolframscript's generator-graph conventions:
// - single center: center first, remaining vertices in vertex-list order;
//   multiple centers: the centers (given order), then each center's
//   BFS layers in vertex-list order, deduplicated
// - edges: each center's incident edges first (center endpoint first,
//   other endpoints ascending), then the remaining edges in canonical
//   ascending order
// - unknown centers are ignored; radius 0 keeps only the centers
// wolframscript's edge order when the neighborhood covers the whole graph
// (and for explicitly-constructed graphs) is an igraph traversal artifact
// that is not replicated.
pub fn neighborhood_graph_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "NeighborhoodGraph".to_string(),
    args: args.to_vec().into(),
  };
  let (vertices, edge_exprs) = match &args[0] {
    Expr::FunctionCall {
      name: gname,
      args: gargs,
    } if gname == "Graph" && gargs.len() >= 2 => match (&gargs[0], &gargs[1]) {
      (Expr::List(v), Expr::List(e)) => (v, e),
      _ => return Ok(unevaluated()),
    },
    _ => return Ok(unevaluated()),
  };
  let radius: i128 = if args.len() == 3 {
    match &args[2] {
      Expr::Integer(r) if *r >= 0 => *r,
      _ => return Ok(unevaluated()),
    }
  } else {
    1
  };
  let vkeys: Vec<String> = vertices.iter().map(expr_to_string).collect();
  let n = vkeys.len();
  let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
  for e in edge_exprs.iter() {
    if let Expr::FunctionCall {
      name: ename,
      args: eargs,
    } = e
      && ename == "UndirectedEdge"
      && eargs.len() == 2
    {
      let a = vkeys.iter().position(|k| *k == expr_to_string(&eargs[0]));
      let b = vkeys.iter().position(|k| *k == expr_to_string(&eargs[1]));
      match (a, b) {
        (Some(a), Some(b)) if a != b => {
          adj[a].push(b);
          adj[b].push(a);
        }
        (Some(_), Some(_)) => {}
        _ => return Ok(unevaluated()),
      }
    } else {
      return Ok(unevaluated());
    }
  }
  for nb in adj.iter_mut() {
    nb.sort_unstable();
    nb.dedup();
  }
  // Centers: a list or a single vertex; unknown centers are ignored
  let requested: Vec<Expr> = match &args[1] {
    Expr::List(items) => items.iter().cloned().collect(),
    single => vec![single.clone()],
  };
  let mut centers: Vec<usize> = Vec::new();
  for r in &requested {
    let key = expr_to_string(r);
    if let Some(i) = vkeys.iter().position(|k| *k == key)
      && !centers.contains(&i)
    {
      centers.push(i);
    }
  }
  // BFS from each center, building the kept vertex order: centers first,
  // then per-center layers in ascending vertex order
  let mut order: Vec<usize> = centers.clone();
  let mut kept = vec![false; n];
  for &c in &centers {
    kept[c] = true;
  }
  for &c in &centers {
    let mut frontier = vec![c];
    for _ in 0..radius {
      let mut next = Vec::new();
      for &u in &frontier {
        for &w in &adj[u] {
          if !kept[w] {
            kept[w] = true;
            next.push(w);
          }
        }
      }
      next.sort_unstable();
      order.extend(&next);
      frontier = next;
    }
  }
  // Single center: remaining vertices in vertex-list order
  if centers.len() == 1 {
    let mut rest: Vec<usize> = order[1..].to_vec();
    rest.sort_unstable();
    order.truncate(1);
    order.extend(rest);
  }

  // Edges: per-center incident edges first, then the rest sorted
  let mut emitted: std::collections::BTreeSet<(usize, usize)> =
    std::collections::BTreeSet::new();
  let mut edges: Vec<Expr> = Vec::new();
  let mk_edge = |a: usize, b: usize| Expr::FunctionCall {
    name: "UndirectedEdge".to_string(),
    args: vec![vertices[a].clone(), vertices[b].clone()].into(),
  };
  for &c in &centers {
    for &w in &adj[c] {
      if kept[w] {
        let key = (c.min(w), c.max(w));
        if emitted.insert(key) {
          edges.push(mk_edge(c, w));
        }
      }
    }
  }
  let mut rest_pairs: Vec<(usize, usize)> = Vec::new();
  for u in 0..n {
    if !kept[u] {
      continue;
    }
    for &w in &adj[u] {
      if u < w && kept[w] && !emitted.contains(&(u, w)) {
        rest_pairs.push((u, w));
      }
    }
  }
  rest_pairs.sort_unstable();
  rest_pairs.dedup();
  for (a, b) in rest_pairs {
    edges.push(mk_edge(a, b));
  }

  Ok(Expr::FunctionCall {
    name: "Graph".to_string(),
    args: vec![
      Expr::List(
        order
          .into_iter()
          .map(|i| vertices[i].clone())
          .collect::<Vec<_>>()
          .into(),
      ),
      Expr::List(edges.into()),
    ]
    .into(),
  })
}

// ---------------------------------------------------------------------------
// Boolean graph predicates: HamiltonianGraphQ, BipartiteGraphQ,
// CompleteGraphQ, LoopFreeGraphQ, PathGraphQ, EmptyGraphQ, SimpleGraphQ.
// All return False for non-graph arguments (matching wolframscript).
// Note: wolframscript's LoopFreeGraphQ[CycleGraph[1]] is True even though
// the displayed edge list contains the self-loop 1<->1 (a provenance quirk
// of its internal representation); Woxi's literal graph gives False there.

/// Parse a Graph expression into (vertex count, edge index pairs including
/// duplicates and self-loops). Returns None for non-graphs.
fn parse_graph_pairs(expr: &Expr) -> Option<(usize, Vec<(usize, usize)>)> {
  let (vertices, edge_exprs) = match expr {
    Expr::FunctionCall {
      name: gname,
      args: gargs,
    } if gname == "Graph" && gargs.len() >= 2 => match (&gargs[0], &gargs[1]) {
      (Expr::List(v), Expr::List(e)) => (v, e),
      _ => return None,
    },
    _ => return None,
  };
  let vkeys: Vec<String> = vertices.iter().map(expr_to_string).collect();
  let mut pairs = Vec::with_capacity(edge_exprs.len());
  for e in edge_exprs.iter() {
    if let Expr::FunctionCall {
      name: ename,
      args: eargs,
    } = e
      && ename == "UndirectedEdge"
      && eargs.len() == 2
    {
      let a = vkeys.iter().position(|k| *k == expr_to_string(&eargs[0]))?;
      let b = vkeys.iter().position(|k| *k == expr_to_string(&eargs[1]))?;
      pairs.push((a, b));
    } else {
      return None;
    }
  }
  Some((vkeys.len(), pairs))
}

fn bool_expr(b: bool) -> Expr {
  Expr::Identifier(if b { "True" } else { "False" }.to_string())
}

pub fn graph_predicate_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let (n, pairs) = match parse_graph_pairs(&args[0]) {
    Some(g) => g,
    None => return Ok(bool_expr(false)),
  };
  let has_loop = pairs.iter().any(|&(a, b)| a == b);
  let mut simple_pairs: Vec<(usize, usize)> = pairs
    .iter()
    .filter(|&&(a, b)| a != b)
    .map(|&(a, b)| (a.min(b), a.max(b)))
    .collect();
  simple_pairs.sort_unstable();
  let multi = {
    let mut s = simple_pairs.clone();
    s.dedup();
    s.len() != simple_pairs.len()
  };
  simple_pairs.dedup();
  let adj = |s: &[(usize, usize)]| {
    let mut a: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(x, y) in s {
      a[x].push(y);
      a[y].push(x);
    }
    a
  };

  let result = match name {
    "EmptyGraphQ" => pairs.is_empty(),
    "LoopFreeGraphQ" => !has_loop,
    "SimpleGraphQ" => !has_loop && !multi,
    "CompleteGraphQ" => {
      // Every pair adjacent exactly once, no loops (wolframscript:
      // a doubled-edge K2 is not complete)
      !has_loop && !multi && simple_pairs.len() == n * n.saturating_sub(1) / 2
    }
    "PathGraphQ" => {
      // A simple connected graph with all degrees <= 2 — wolframscript
      // counts cycles as path graphs. The null graph is not a path.
      if n == 0 || has_loop || multi {
        false
      } else if n == 1 {
        pairs.is_empty()
      } else {
        let a = adj(&simple_pairs);
        (0..n).all(|v| a[v].len() <= 2) && connected_from(&a, 0) == n
      }
    }
    "BipartiteGraphQ" => {
      // 2-colorable (loops make it odd-cyclic)
      if has_loop {
        false
      } else {
        let a = adj(&simple_pairs);
        let mut color = vec![-1i8; n];
        let mut ok = true;
        'outer: for start in 0..n {
          if color[start] != -1 {
            continue;
          }
          color[start] = 0;
          let mut queue = std::collections::VecDeque::from([start]);
          while let Some(u) = queue.pop_front() {
            for &w in &a[u] {
              if color[w] == -1 {
                color[w] = 1 - color[u];
                queue.push_back(w);
              } else if color[w] == color[u] {
                ok = false;
                break 'outer;
              }
            }
          }
        }
        ok
      }
    }
    "HamiltonianGraphQ" => {
      // wolframscript: K1 (even with a self-loop) and the doubled-edge
      // 2-cycle count as Hamiltonian; the null graph does not
      if n == 0 {
        false
      } else if n == 1 {
        true
      } else if n == 2 {
        // Needs two parallel edges to form the 2-cycle
        pairs.iter().filter(|&&(a, b)| a != b).count() >= 2
      } else {
        let a = adj(&simple_pairs);
        if a.iter().any(|nb| nb.len() < 2) || connected_from(&a, 0) != n {
          false
        } else {
          let mut visited = vec![false; n];
          visited[0] = true;
          hamiltonian_dfs(&a, 0, 1, &mut visited)
        }
      }
    }
    _ => false,
  };
  let _ = args;
  Ok(bool_expr(result))
}

/// Number of vertices reachable from `start`.
fn connected_from(adj: &[Vec<usize>], start: usize) -> usize {
  let n = adj.len();
  let mut seen = vec![false; n];
  seen[start] = true;
  let mut queue = std::collections::VecDeque::from([start]);
  let mut count = 1;
  while let Some(u) = queue.pop_front() {
    for &w in &adj[u] {
      if !seen[w] {
        seen[w] = true;
        count += 1;
        queue.push_back(w);
      }
    }
  }
  count
}

fn hamiltonian_dfs(
  adj: &[Vec<usize>],
  current: usize,
  depth: usize,
  visited: &mut [bool],
) -> bool {
  let n = adj.len();
  if depth == n {
    return adj[current].contains(&0);
  }
  for &w in &adj[current] {
    if !visited[w] {
      visited[w] = true;
      if hamiltonian_dfs(adj, w, depth + 1, visited) {
        return true;
      }
      visited[w] = false;
    }
  }
  false
}

// ---------------------------------------------------------------------------
// PlanarGraphQ — planarity testing via the Demoucron-Malgrange-Pertuiset
// (DMP) face-embedding algorithm, run per biconnected component.
// Self-loops, parallel edges, bridges, and components with < 5 vertices
// never affect planarity.

/// Biconnected components (as edge lists) of a simple graph.
fn biconnected_components(
  n: usize,
  adj: &[Vec<usize>],
) -> Vec<Vec<(usize, usize)>> {
  let mut comps = Vec::new();
  let mut disc = vec![0usize; n];
  let mut low = vec![0usize; n];
  let mut timer = 1usize;
  let mut edge_stack: Vec<(usize, usize)> = Vec::new();
  for start in 0..n {
    if disc[start] != 0 {
      continue;
    }
    let mut stack: Vec<(usize, usize, usize)> = vec![(start, usize::MAX, 0)];
    disc[start] = timer;
    low[start] = timer;
    timer += 1;
    while let Some(&mut (u, parent, ref mut i)) = stack.last_mut() {
      if *i < adj[u].len() {
        let w = adj[u][*i];
        *i += 1;
        if disc[w] == 0 {
          edge_stack.push((u, w));
          disc[w] = timer;
          low[w] = timer;
          timer += 1;
          stack.push((w, u, 0));
        } else if w != parent && disc[w] < disc[u] {
          edge_stack.push((u, w));
          low[u] = low[u].min(disc[w]);
        }
      } else {
        stack.pop();
        if let Some(&mut (p, _, _)) = stack.last_mut() {
          low[p] = low[p].min(low[u]);
          if low[u] >= disc[p] {
            let mut comp = Vec::new();
            while let Some(&(a, b)) = edge_stack.last() {
              comp.push((a, b));
              edge_stack.pop();
              if a == p && b == u {
                break;
              }
            }
            if !comp.is_empty() {
              comps.push(comp);
            }
          }
        }
      }
    }
  }
  comps
}

/// DMP planarity test for one biconnected component given as an edge list.
fn dmp_planar(edges: &[(usize, usize)]) -> bool {
  let mut verts: Vec<usize> = edges.iter().flat_map(|&(a, b)| [a, b]).collect();
  verts.sort_unstable();
  verts.dedup();
  let k = verts.len();
  // Fewer than 5 vertices, or too few edges to contain K5/K3,3
  if k < 5 || edges.len() < 9 {
    return true;
  }
  if edges.len() > 3 * k - 6 {
    return false;
  }
  let index = |v: usize| verts.binary_search(&v).unwrap();
  let es: Vec<(usize, usize)> =
    edges.iter().map(|&(a, b)| (index(a), index(b))).collect();
  let mut adj: Vec<Vec<usize>> = vec![Vec::new(); k];
  for &(a, b) in &es {
    adj[a].push(b);
    adj[b].push(a);
  }

  // Initial cycle via an iterative DFS back edge
  let mut parent = vec![usize::MAX; k];
  let mut state = vec![0u8; k];
  let mut dfs_stack = vec![(0usize, 0usize)];
  state[0] = 1;
  let mut back_edge: Option<(usize, usize)> = None;
  while let Some(&mut (u, ref mut i)) = dfs_stack.last_mut() {
    if *i < adj[u].len() {
      let w = adj[u][*i];
      *i += 1;
      if state[w] == 0 {
        state[w] = 1;
        parent[w] = u;
        dfs_stack.push((w, 0));
      } else if state[w] == 1 && w != parent[u] {
        back_edge = Some((u, w));
        break;
      }
    } else {
      state[u] = 2;
      dfs_stack.pop();
    }
  }
  let (mut cu, cw) = match back_edge {
    Some(e) => e,
    None => return true, // acyclic — cannot happen in a bicomp, but safe
  };
  let mut cycle = vec![cu];
  while cu != cw {
    cu = parent[cu];
    cycle.push(cu);
  }

  let mut in_h = vec![false; k];
  let mut h_edges: std::collections::BTreeSet<(usize, usize)> =
    Default::default();
  for &v in &cycle {
    in_h[v] = true;
  }
  let clen = cycle.len();
  for i in 0..clen {
    let (a, b) = (cycle[i], cycle[(i + 1) % clen]);
    h_edges.insert((a.min(b), a.max(b)));
  }
  // The initial cycle bounds two faces
  let mut faces: Vec<Vec<usize>> = vec![cycle.clone(), cycle];

  while h_edges.len() < es.len() {
    // Fragments relative to H: chords between H-vertices, and connected
    // components of G - V(H) together with their attachment edges.
    // Each fragment carries one embeddable path between two attachments.
    let mut fragments: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    for &(a, b) in &es {
      if in_h[a] && in_h[b] && !h_edges.contains(&(a.min(b), a.max(b))) {
        fragments.push((vec![a.min(b), a.max(b)], vec![a, b]));
      }
    }
    let mut comp_id = vec![usize::MAX; k];
    let mut n_comps = 0usize;
    for v in 0..k {
      if in_h[v] || comp_id[v] != usize::MAX {
        continue;
      }
      comp_id[v] = n_comps;
      let mut queue = std::collections::VecDeque::from([v]);
      while let Some(u) = queue.pop_front() {
        for &w in &adj[u] {
          if !in_h[w] && comp_id[w] == usize::MAX {
            comp_id[w] = n_comps;
            queue.push_back(w);
          }
        }
      }
      n_comps += 1;
    }
    for c in 0..n_comps {
      let members: Vec<usize> = (0..k).filter(|&v| comp_id[v] == c).collect();
      let mut attachments: Vec<usize> = members
        .iter()
        .flat_map(|&u| adj[u].iter().copied().filter(|&w| in_h[w]))
        .collect();
      attachments.sort_unstable();
      attachments.dedup();
      // In a biconnected graph every fragment has >= 2 attachments.
      let a0 = attachments[0];
      let a1 = attachments[1];
      // BFS through the component from a neighbor of a0 to a neighbor of a1
      let mut prev = vec![usize::MAX; k];
      let mut queue = std::collections::VecDeque::new();
      for &u in &members {
        if adj[u].contains(&a0) {
          prev[u] = a0;
          queue.push_back(u);
        }
      }
      let mut endpoint = usize::MAX;
      while let Some(u) = queue.pop_front() {
        if adj[u].contains(&a1) {
          endpoint = u;
          break;
        }
        for &w in &adj[u] {
          if !in_h[w] && comp_id[w] == c && prev[w] == usize::MAX {
            prev[w] = u;
            queue.push_back(w);
          }
        }
      }
      if endpoint == usize::MAX {
        // a0's and a1's component neighborhoods did not connect — should
        // not happen inside one component, but bail out conservatively
        return false;
      }
      let mut path = vec![a1];
      let mut cur = endpoint;
      while cur != a0 {
        path.push(cur);
        cur = prev[cur];
      }
      path.push(a0);
      fragments.push((attachments, path));
    }

    // Admissible faces per fragment; embed the most constrained fragment
    let mut best: Option<(usize, usize, usize)> = None; // (count, frag, face)
    for (fi, (attachments, _)) in fragments.iter().enumerate() {
      let mut count = 0usize;
      let mut first_face = usize::MAX;
      for (fj, f) in faces.iter().enumerate() {
        if attachments.iter().all(|a| f.contains(a)) {
          count += 1;
          if first_face == usize::MAX {
            first_face = fj;
          }
        }
      }
      if count == 0 {
        return false;
      }
      if best.is_none_or(|(c, _, _)| count < c) {
        best = Some((count, fi, first_face));
      }
    }
    let (_, fi, face_idx) = match best {
      Some(b) => b,
      None => return true, // no fragments left (unreachable: loop guard)
    };
    let path = fragments[fi].1.clone();

    // Add the path to H
    for w in path.windows(2) {
      h_edges.insert((w[0].min(w[1]), w[0].max(w[1])));
    }
    for &v in &path {
      in_h[v] = true;
    }
    // Split the face along the path
    let face = faces.swap_remove(face_idx);
    let pa = path[0];
    let pb = *path.last().unwrap();
    let ia = face.iter().position(|&v| v == pa).unwrap();
    let ib = face.iter().position(|&v| v == pb).unwrap();
    let len = face.len();
    let interior: Vec<usize> = path[1..path.len() - 1].to_vec();
    // Arc from pa forward to pb, plus the path interior walking back
    let mut f1 = Vec::new();
    let mut t = ia;
    loop {
      f1.push(face[t]);
      if t == ib {
        break;
      }
      t = (t + 1) % len;
    }
    f1.extend(interior.iter().rev().copied());
    // Arc from pb forward to pa, plus the path interior walking forward
    let mut f2 = Vec::new();
    let mut t = ib;
    loop {
      f2.push(face[t]);
      if t == ia {
        break;
      }
      t = (t + 1) % len;
    }
    f2.extend(interior.iter().copied());
    faces.push(f1);
    faces.push(f2);
  }
  true
}

/// PlanarGraphQ[g]
pub fn planar_graph_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (n, pairs) = match parse_graph_pairs(&args[0]) {
    Some(g) => g,
    None => return Ok(bool_expr(false)),
  };
  // Reduce to the underlying simple graph
  let mut simple: Vec<(usize, usize)> = pairs
    .iter()
    .filter(|&&(a, b)| a != b)
    .map(|&(a, b)| (a.min(b), a.max(b)))
    .collect();
  simple.sort_unstable();
  simple.dedup();
  let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
  for &(a, b) in &simple {
    adj[a].push(b);
    adj[b].push(a);
  }
  let planar = biconnected_components(n, &adj)
    .iter()
    .all(|comp| dmp_planar(comp));
  Ok(bool_expr(planar))
}

// ---------------------------------------------------------------------------
// Graph metrics: GlobalClusteringCoefficient, MeanClusteringCoefficient,
// GraphDensity, MeanGraphDistance. All exact rationals (or Infinity for
// the mean distance of a disconnected graph); non-graphs and graphs with
// fewer than two vertices (where the formulas degenerate) stay unevaluated.

fn make_rational(num: i128, den: i128) -> Expr {
  Expr::FunctionCall {
    name: "Rational".to_string(),
    args: vec![Expr::Integer(num), Expr::Integer(den)].into(),
  }
}

/// GraphDensity for a graph containing directed edges: the fraction of the
/// n(n-1) possible ordered vertex pairs that are edges (distinct edges, no
/// self-loops). Returns `None` unless every edge is directed and the graph has
/// at least two vertices.
fn directed_graph_density(expr: &Expr) -> Option<Expr> {
  let (vertices, edge_exprs) = match expr {
    Expr::FunctionCall { name, args } if name == "Graph" && args.len() >= 2 => {
      match (&args[0], &args[1]) {
        (Expr::List(v), Expr::List(e)) => (v, e),
        _ => return None,
      }
    }
    _ => return None,
  };
  let n = vertices.len();
  if n <= 1 {
    return None;
  }
  let index: std::collections::HashMap<String, usize> = vertices
    .iter()
    .enumerate()
    .map(|(i, v)| (expr_to_string(v), i))
    .collect();
  let mut edges: std::collections::HashSet<(usize, usize)> =
    std::collections::HashSet::new();
  for e in edge_exprs.iter() {
    let Expr::FunctionCall { name, args } = e else {
      return None;
    };
    if name != "DirectedEdge" || args.len() != 2 {
      return None; // undirected/mixed handled by the caller
    }
    let (Some(&a), Some(&b)) = (
      index.get(&expr_to_string(&args[0])),
      index.get(&expr_to_string(&args[1])),
    ) else {
      return None;
    };
    if a != b {
      edges.insert((a, b));
    }
  }
  let m = edges.len() as i128;
  Some(
    crate::evaluator::evaluate_expr_to_expr(&make_rational(
      m,
      (n as i128) * (n as i128 - 1),
    ))
    .unwrap_or_else(|_| Expr::Integer(0)),
  )
}

/// MeanGraphDistance for a graph containing directed edges: the mean directed
/// shortest-path distance over all n(n-1) ordered vertex pairs, or Infinity if
/// any pair is unreachable. Directed edges point one way; undirected edges
/// point both ways (so mixed graphs work too). Returns `None` for anything
/// that is not such a graph.
fn directed_mean_graph_distance(expr: &Expr) -> Option<Expr> {
  let (vertices, edge_exprs) = match expr {
    Expr::FunctionCall { name, args } if name == "Graph" && args.len() >= 2 => {
      match (&args[0], &args[1]) {
        (Expr::List(v), Expr::List(e)) => (v, e),
        _ => return None,
      }
    }
    _ => return None,
  };
  let n = vertices.len();
  if n <= 1 {
    return None;
  }
  let index: std::collections::HashMap<String, usize> = vertices
    .iter()
    .enumerate()
    .map(|(i, v)| (expr_to_string(v), i))
    .collect();
  let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
  for e in edge_exprs.iter() {
    let Expr::FunctionCall { name, args } = e else {
      return None;
    };
    if args.len() != 2 {
      return None;
    }
    let (Some(&a), Some(&b)) = (
      index.get(&expr_to_string(&args[0])),
      index.get(&expr_to_string(&args[1])),
    ) else {
      return None;
    };
    match name.as_str() {
      "DirectedEdge" => adj[a].push(b),
      "UndirectedEdge" => {
        adj[a].push(b);
        adj[b].push(a);
      }
      _ => return None,
    }
  }
  let mut total: i128 = 0;
  for start in 0..n {
    let mut dist = vec![usize::MAX; n];
    dist[start] = 0;
    let mut queue = std::collections::VecDeque::from([start]);
    let mut reached = 1usize;
    while let Some(u) = queue.pop_front() {
      for &w in &adj[u] {
        if dist[w] == usize::MAX {
          dist[w] = dist[u] + 1;
          reached += 1;
          queue.push_back(w);
        }
      }
    }
    if reached != n {
      return Some(Expr::Identifier("Infinity".to_string()));
    }
    total += dist.iter().map(|&d| d as i128).sum::<i128>();
  }
  Some(
    crate::evaluator::evaluate_expr_to_expr(&make_rational(
      total,
      (n as i128) * (n as i128 - 1),
    ))
    .unwrap_or(Expr::Integer(0)),
  )
}

/// GlobalClusteringCoefficient for a graph containing directed edges: the
/// directed transitivity trace(A^3) / sum_{i != k} (A^2)[i][k] -- the fraction
/// of directed length-2 paths i->j->k (distinct endpoints) that are closed by
/// an edge k->i. Directed edges point one way; undirected edges both ways.
/// Returns `None` for anything that is not such a graph.
fn directed_global_clustering(expr: &Expr) -> Option<Expr> {
  let (vertices, edge_exprs) = match expr {
    Expr::FunctionCall { name, args } if name == "Graph" && args.len() >= 2 => {
      match (&args[0], &args[1]) {
        (Expr::List(v), Expr::List(e)) => (v, e),
        _ => return None,
      }
    }
    _ => return None,
  };
  let n = vertices.len();
  if n == 0 {
    return None;
  }
  let index: std::collections::HashMap<String, usize> = vertices
    .iter()
    .enumerate()
    .map(|(i, v)| (expr_to_string(v), i))
    .collect();
  // Simple 0/1 directed adjacency, self-loops excluded.
  let mut a = vec![vec![0i128; n]; n];
  for e in edge_exprs.iter() {
    let Expr::FunctionCall { name, args } = e else {
      return None;
    };
    if args.len() != 2 {
      return None;
    }
    let (Some(&u), Some(&v)) = (
      index.get(&expr_to_string(&args[0])),
      index.get(&expr_to_string(&args[1])),
    ) else {
      return None;
    };
    match name.as_str() {
      "DirectedEdge" => {
        if u != v {
          a[u][v] = 1;
        }
      }
      "UndirectedEdge" => {
        if u != v {
          a[u][v] = 1;
          a[v][u] = 1;
        }
      }
      _ => return None,
    }
  }
  let matmul = |x: &[Vec<i128>], y: &[Vec<i128>]| -> Vec<Vec<i128>> {
    let mut r = vec![vec![0i128; n]; n];
    for i in 0..n {
      for k in 0..n {
        if x[i][k] != 0 {
          for j in 0..n {
            r[i][j] += x[i][k] * y[k][j];
          }
        }
      }
    }
    r
  };
  let a2 = matmul(&a, &a);
  let a3 = matmul(&a2, &a);
  let num: i128 = (0..n).map(|i| a3[i][i]).sum();
  let den: i128 = (0..n)
    .flat_map(|i| (0..n).map(move |k| (i, k)))
    .filter(|(i, k)| i != k)
    .map(|(i, k)| a2[i][k])
    .sum();
  if den == 0 {
    return Some(Expr::Integer(0));
  }
  Some(
    crate::evaluator::evaluate_expr_to_expr(&make_rational(num, den))
      .unwrap_or(Expr::Integer(0)),
  )
}

/// Directed local clustering coefficient per vertex, as (closed, total) pairs.
/// For vertex v, `total` counts the in->v->out paths (i, k) with distinct
/// endpoints i != k and `closed` those additionally closed by an arc k -> i (a
/// directed 3-cycle). Undirected edges count both ways. Returns `None` unless
/// the graph contains a directed edge.
pub fn directed_local_clustering(expr: &Expr) -> Option<Vec<(i128, i128)>> {
  let (vertices, edge_exprs) = match expr {
    Expr::FunctionCall { name, args } if name == "Graph" && args.len() >= 2 => {
      match (&args[0], &args[1]) {
        (Expr::List(v), Expr::List(e)) => (v, e),
        _ => return None,
      }
    }
    _ => return None,
  };
  let n = vertices.len();
  let index: std::collections::HashMap<String, usize> = vertices
    .iter()
    .enumerate()
    .map(|(i, v)| (expr_to_string(v), i))
    .collect();
  let mut in_nbrs: Vec<std::collections::HashSet<usize>> =
    vec![std::collections::HashSet::new(); n];
  let mut out_nbrs: Vec<std::collections::HashSet<usize>> =
    vec![std::collections::HashSet::new(); n];
  let mut arcs: std::collections::HashSet<(usize, usize)> =
    std::collections::HashSet::new();
  let mut directed = false;
  for e in edge_exprs.iter() {
    let Expr::FunctionCall { name, args } = e else {
      return None;
    };
    if args.len() != 2 {
      return None;
    }
    let (Some(a), Some(b)) = (
      index.get(&expr_to_string(&args[0])).copied(),
      index.get(&expr_to_string(&args[1])).copied(),
    ) else {
      return None;
    };
    let dirs: &[(usize, usize)] = match name.as_str() {
      "DirectedEdge" => {
        directed = true;
        &[(a, b)]
      }
      "UndirectedEdge" => &[(a, b), (b, a)],
      _ => return None,
    };
    for &(x, y) in dirs {
      if x != y {
        out_nbrs[x].insert(y);
        in_nbrs[y].insert(x);
        arcs.insert((x, y));
      }
    }
  }
  if !directed {
    return None;
  }
  Some(
    (0..n)
      .map(|v| {
        let mut total = 0i128;
        let mut closed = 0i128;
        for &i in &in_nbrs[v] {
          for &k in &out_nbrs[v] {
            if i != k {
              total += 1;
              if arcs.contains(&(k, i)) {
                closed += 1;
              }
            }
          }
        }
        (closed, total)
      })
      .collect(),
  )
}

/// MeanDegreeConnectivity for a graph containing directed edges: for each
/// degree k = 0, 1, 2, ..., the mean over vertices of (total, in+out) degree k
/// of the average degree of their neighbours. Every edge contributes one to
/// each endpoint's degree and makes the endpoints neighbours (in or out alike).
/// Returns `None` for anything that is not such a graph.
fn directed_mean_degree_connectivity(expr: &Expr) -> Option<Expr> {
  let (vertices, edge_exprs) = match expr {
    Expr::FunctionCall { name, args } if name == "Graph" && args.len() >= 2 => {
      match (&args[0], &args[1]) {
        (Expr::List(v), Expr::List(e)) => (v, e),
        _ => return None,
      }
    }
    _ => return None,
  };
  let n = vertices.len();
  if n == 0 {
    return None;
  }
  let index: std::collections::HashMap<String, usize> = vertices
    .iter()
    .enumerate()
    .map(|(i, v)| (expr_to_string(v), i))
    .collect();
  let mut deg = vec![0i128; n];
  let mut nbr_of: Vec<Vec<usize>> = vec![Vec::new(); n];
  for e in edge_exprs.iter() {
    let Expr::FunctionCall { name, args } = e else {
      return None;
    };
    if (name != "DirectedEdge" && name != "UndirectedEdge") || args.len() != 2 {
      return None;
    }
    let (Some(a), Some(b)) = (
      index.get(&expr_to_string(&args[0])).copied(),
      index.get(&expr_to_string(&args[1])).copied(),
    ) else {
      return None;
    };
    deg[a] += 1;
    deg[b] += 1;
    nbr_of[a].push(b);
    nbr_of[b].push(a);
  }
  let maxdeg = deg.iter().copied().max().unwrap_or(0);
  // Sum of neighbour degrees for each vertex.
  let s: Vec<i128> = (0..n)
    .map(|v| nbr_of[v].iter().map(|&u| deg[u]).sum())
    .collect();
  let mut result = Vec::with_capacity((maxdeg + 1) as usize);
  for k in 0..=maxdeg {
    let verts: Vec<usize> = (0..n).filter(|&v| deg[v] == k).collect();
    if k == 0 || verts.is_empty() {
      result.push(Expr::Integer(0));
    } else {
      let sum_s: i128 = verts.iter().map(|&v| s[v]).sum();
      let count = verts.len() as i128;
      result.push(
        crate::evaluator::evaluate_expr_to_expr(&make_rational(
          sum_s,
          k * count,
        ))
        .unwrap_or(Expr::Integer(0)),
      );
    }
  }
  Some(Expr::List(result.into()))
}

pub fn graph_metric_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: name.to_string(),
    args: args.to_vec().into(),
  };
  // GraphLinkEfficiency = 1 - MeanGraphDistance/EdgeCount (decoded from
  // wolframscript's exact rationals; disconnected graphs give -Infinity
  // through the infinite mean distance).
  if name == "GraphLinkEfficiency" {
    let mgd = graph_metric_ast("MeanGraphDistance", args)?;
    if matches!(&mgd, Expr::FunctionCall { name, .. } if name == "MeanGraphDistance")
    {
      return Ok(unevaluated());
    }
    let ec = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "EdgeCount".to_string(),
      args: vec![args[0].clone()].into(),
    })?;
    if !matches!(&ec, Expr::Integer(m) if *m > 0) {
      return Ok(unevaluated());
    }
    return crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(Expr::Integer(1)),
      right: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(mgd),
        right: Box::new(ec),
      }),
    });
  }
  let (n, pairs) = match parse_graph_pairs(&args[0]) {
    Some(g) => g,
    None => {
      // parse_graph_pairs only accepts undirected edges; handle the
      // directed case for the metrics that have a directed definition.
      if name == "GraphDensity"
        && let Some(result) = directed_graph_density(&args[0])
      {
        return Ok(result);
      }
      if name == "MeanGraphDistance"
        && let Some(result) = directed_mean_graph_distance(&args[0])
      {
        return Ok(result);
      }
      if name == "GlobalClusteringCoefficient"
        && let Some(result) = directed_global_clustering(&args[0])
      {
        return Ok(result);
      }
      if name == "MeanDegreeConnectivity"
        && let Some(result) = directed_mean_degree_connectivity(&args[0])
      {
        return Ok(result);
      }
      if name == "MeanClusteringCoefficient"
        && let Some(local) = directed_local_clustering(&args[0])
        && !local.is_empty()
      {
        // Mean of the per-vertex directed local clustering coefficients.
        let terms: Vec<Expr> = local
          .iter()
          .map(|&(c, t)| {
            if t == 0 {
              Expr::Integer(0)
            } else {
              make_rational(c, t)
            }
          })
          .collect();
        let sum = Expr::FunctionCall {
          name: "Plus".to_string(),
          args: terms.into(),
        };
        let mean = Expr::FunctionCall {
          name: "Divide".to_string(),
          args: vec![sum, Expr::Integer(local.len() as i128)].into(),
        };
        return Ok(
          crate::evaluator::evaluate_expr_to_expr(&mean)
            .unwrap_or_else(|_| unevaluated()),
        );
      }
      return Ok(unevaluated());
    }
  };
  // Underlying simple graph
  let mut simple: Vec<(usize, usize)> = pairs
    .iter()
    .filter(|&&(a, b)| a != b)
    .map(|&(a, b)| (a.min(b), a.max(b)))
    .collect();
  simple.sort_unstable();
  simple.dedup();
  let m = simple.len() as i128;
  let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
  for &(a, b) in &simple {
    adj[a].push(b);
    adj[b].push(a);
  }
  let is_adjacent =
    |a: usize, b: usize| simple.binary_search(&(a.min(b), a.max(b))).is_ok();

  let evaluated = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e);
  match name {
    "GraphDensity" => {
      if n <= 1 {
        return Ok(unevaluated());
      }
      evaluated(make_rational(2 * m, (n as i128) * (n as i128 - 1)))
    }
    "GlobalClusteringCoefficient" => {
      let triples: i128 = adj
        .iter()
        .map(|nb| {
          let d = nb.len() as i128;
          d * (d - 1) / 2
        })
        .sum();
      if triples == 0 {
        return Ok(Expr::Integer(0));
      }
      let mut triangles: i128 = 0;
      for &(a, b) in &simple {
        for &c in &adj[a] {
          if c > b && is_adjacent(b, c) {
            triangles += 1;
          }
        }
      }
      evaluated(make_rational(3 * triangles, triples))
    }
    "MeanClusteringCoefficient" => {
      if n == 0 {
        return Ok(unevaluated());
      }
      // Sum of local coefficients as one rational: each vertex
      // contributes links_v / C(deg_v, 2) (0 when deg_v < 2)
      let mut num: i128 = 0;
      let mut den: i128 = 1;
      for v in 0..n {
        let d = adj[v].len() as i128;
        if d < 2 {
          continue;
        }
        let mut links: i128 = 0;
        for i in 0..adj[v].len() {
          for j in (i + 1)..adj[v].len() {
            if is_adjacent(adj[v][i], adj[v][j]) {
              links += 1;
            }
          }
        }
        let vden = d * (d - 1) / 2;
        num = num * vden + links * den;
        den *= vden;
      }
      evaluated(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![make_rational(num, den), make_rational(1, n as i128)].into(),
      })
    }
    "MeanDegreeConnectivity" => {
      if n == 0 {
        return Ok(unevaluated());
      }
      // Degree of every vertex in the underlying simple graph.
      let deg: Vec<i128> = adj.iter().map(|nb| nb.len() as i128).collect();
      let max_deg = deg.iter().copied().max().unwrap_or(0);
      // Result is indexed by degree k = 0..maxDeg; entry k is the mean degree
      // of the neighbors of all degree-k vertices: the sum of neighbor
      // degrees over those vertices divided by the sum of their degrees
      // (0 when there are no degree-k vertices).
      let mut result = Vec::with_capacity((max_deg + 1) as usize);
      for k in 0..=max_deg {
        let mut num: i128 = 0;
        let mut den: i128 = 0;
        for v in 0..n {
          if deg[v] == k {
            den += deg[v];
            for &w in &adj[v] {
              num += deg[w];
            }
          }
        }
        if den == 0 {
          result.push(Expr::Integer(0));
        } else {
          result.push(evaluated(make_rational(num, den))?);
        }
      }
      Ok(Expr::List(result.into()))
    }
    "MeanGraphDistance" => {
      if n <= 1 {
        return Ok(unevaluated());
      }
      let mut total: i128 = 0;
      for start in 0..n {
        let mut dist = vec![usize::MAX; n];
        dist[start] = 0;
        let mut queue = std::collections::VecDeque::from([start]);
        let mut reached = 1usize;
        while let Some(u) = queue.pop_front() {
          for &w in &adj[u] {
            if dist[w] == usize::MAX {
              dist[w] = dist[u] + 1;
              reached += 1;
              queue.push_back(w);
            }
          }
        }
        if reached != n {
          return Ok(Expr::Identifier("Infinity".to_string()));
        }
        total += dist.iter().map(|&d| d as i128).sum::<i128>();
      }
      // total counts ordered pairs; the denominator is n(n-1)
      evaluated(make_rational(total, (n as i128) * (n as i128 - 1)))
    }
    _ => Ok(unevaluated()),
  }
}

// ---------------------------------------------------------------------------
// Graph accessors: AdjacencyList, IncidenceList, EdgeIndex. Undirected
// graphs; invalid vertices/edges emit ::inv, non-graphs stay unevaluated.

pub fn graph_accessor_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: name.to_string(),
    args: args.to_vec().into(),
  };
  let (vertices, edge_exprs) = match &args[0] {
    Expr::FunctionCall {
      name: gname,
      args: gargs,
    } if gname == "Graph" && gargs.len() >= 2 => match (&gargs[0], &gargs[1]) {
      (Expr::List(v), Expr::List(e)) => (v, e),
      _ => return Ok(unevaluated()),
    },
    _ => return Ok(unevaluated()),
  };
  let vkeys: Vec<String> = vertices.iter().map(expr_to_string).collect();
  let n = vkeys.len();
  let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(edge_exprs.len());
  for e in edge_exprs.iter() {
    if let Expr::FunctionCall {
      name: ename,
      args: eargs,
    } = e
      && ename == "UndirectedEdge"
      && eargs.len() == 2
    {
      let a = vkeys.iter().position(|k| *k == expr_to_string(&eargs[0]));
      let b = vkeys.iter().position(|k| *k == expr_to_string(&eargs[1]));
      match (a, b) {
        (Some(a), Some(b)) => pairs.push((a, b)),
        _ => return Ok(unevaluated()),
      }
    } else {
      return Ok(unevaluated());
    }
  }
  let inv = |what: &str, arg: &Expr| {
    crate::emit_message(&format!(
      "{}::inv: The argument {} in {} is not a valid {}.",
      name,
      crate::syntax::format_expr(arg, crate::syntax::ExprForm::Output),
      crate::syntax::format_expr(
        &Expr::FunctionCall {
          name: name.to_string(),
          args: args.to_vec().into(),
        },
        crate::syntax::ExprForm::Output
      ),
      what
    ));
  };
  // Neighbors of one vertex, in vertex-list order, deduplicated
  let neighbors = |v: usize| -> Vec<usize> {
    let mut out: Vec<usize> = pairs
      .iter()
      .filter_map(|&(a, b)| {
        if a == v && b != v {
          Some(b)
        } else if b == v && a != v {
          Some(a)
        } else {
          None
        }
      })
      .collect();
    out.sort_unstable();
    out.dedup();
    out
  };

  match (name, args.len()) {
    ("AdjacencyList", 1) => Ok(Expr::List(
      (0..n)
        .map(|v| {
          Expr::List(
            neighbors(v)
              .into_iter()
              .map(|w| vertices[w].clone())
              .collect::<Vec<_>>()
              .into(),
          )
        })
        .collect::<Vec<_>>()
        .into(),
    )),
    ("AdjacencyList", 2) => {
      let key = expr_to_string(&args[1]);
      let Some(v) = vkeys.iter().position(|k| *k == key) else {
        inv("vertex", &args[1]);
        return Ok(unevaluated());
      };
      Ok(Expr::List(
        neighbors(v)
          .into_iter()
          .map(|w| vertices[w].clone())
          .collect::<Vec<_>>()
          .into(),
      ))
    }
    ("IncidenceList", 2) => {
      let key = expr_to_string(&args[1]);
      let Some(v) = vkeys.iter().position(|k| *k == key) else {
        inv("vertex", &args[1]);
        return Ok(unevaluated());
      };
      Ok(Expr::List(
        pairs
          .iter()
          .zip(edge_exprs.iter())
          .filter(|((a, b), _)| *a == v || *b == v)
          .map(|(_, e)| e.clone())
          .collect::<Vec<_>>()
          .into(),
      ))
    }
    ("EdgeIndex", 2) => {
      // The edge may be given as UndirectedEdge or TwoWayRule, in
      // either endpoint order
      let endpoints = match &args[1] {
        Expr::FunctionCall {
          name: ename,
          args: eargs,
        } if (ename == "UndirectedEdge" || ename == "TwoWayRule")
          && eargs.len() == 2 =>
        {
          Some((expr_to_string(&eargs[0]), expr_to_string(&eargs[1])))
        }
        _ => None,
      };
      let found = endpoints.as_ref().and_then(|(x, y)| {
        let xi = vkeys.iter().position(|k| k == x)?;
        let yi = vkeys.iter().position(|k| k == y)?;
        pairs
          .iter()
          .position(|&(a, b)| (a == xi && b == yi) || (a == yi && b == xi))
      });
      match found {
        Some(i) => Ok(Expr::Integer((i + 1) as i128)),
        None => {
          inv("edge", &args[1]);
          Ok(unevaluated())
        }
      }
    }
    _ => Ok(unevaluated()),
  }
}

/// GraphAssortativity[g] — Newman's degree assortativity coefficient as an
/// exact rational: the Pearson correlation of the endpoint degrees over
/// all edges (both orientations). Regular and edgeless graphs give 0;
/// non-graphs stay unevaluated.
pub fn graph_assortativity_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "GraphAssortativity".to_string(),
    args: args.to_vec().into(),
  };
  let (n, pairs) = match parse_graph_pairs(&args[0]) {
    Some(g) => g,
    None => return Ok(unevaluated()),
  };
  let mut simple: Vec<(usize, usize)> = pairs
    .iter()
    .filter(|&&(a, b)| a != b)
    .map(|&(a, b)| (a.min(b), a.max(b)))
    .collect();
  simple.sort_unstable();
  simple.dedup();
  if simple.is_empty() {
    return Ok(Expr::Integer(0));
  }
  let mut deg = vec![0i128; n];
  for &(a, b) in &simple {
    deg[a] += 1;
    deg[b] += 1;
  }
  // Sums over both edge orientations
  let m2 = 2 * simple.len() as i128;
  let mut s_x: i128 = 0;
  let mut s_xx: i128 = 0;
  let mut s_xy: i128 = 0;
  for &(a, b) in &simple {
    let (da, db) = (deg[a], deg[b]);
    s_x += da + db;
    s_xx += da * da + db * db;
    s_xy += 2 * da * db;
  }
  let num = m2 * s_xy - s_x * s_x;
  let den = m2 * s_xx - s_x * s_x;
  if den == 0 {
    return Ok(Expr::Integer(0));
  }
  crate::evaluator::evaluate_expr_to_expr(&make_rational(num, den))
}
