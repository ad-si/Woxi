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

  // Dijkstra with a binary min-heap (costs are non-negative).
  use std::collections::BinaryHeap;
  let mut dist: HashMap<String, f64> = HashMap::new();
  let mut prev: HashMap<String, String> = HashMap::new();
  let mut heap: BinaryHeap<(
    std::cmp::Reverse<ordered_f64::OrderedF64>,
    String,
  )> = BinaryHeap::new();
  dist.insert(start.clone(), 0.0);
  heap.push((
    std::cmp::Reverse(ordered_f64::OrderedF64(0.0)),
    start.clone(),
  ));
  while let Some((std::cmp::Reverse(ordered_f64::OrderedF64(d)), u)) =
    heap.pop()
  {
    if d > *dist.get(&u).unwrap_or(&f64::INFINITY) {
      continue;
    }
    if u == goal {
      break;
    }
    if let Some(neighbors) = adj.get(&u) {
      for (v, w) in neighbors {
        let nd = d + w;
        if nd < *dist.get(v).unwrap_or(&f64::INFINITY) {
          dist.insert(v.clone(), nd);
          prev.insert(v.clone(), u.clone());
          heap
            .push((std::cmp::Reverse(ordered_f64::OrderedF64(nd)), v.clone()));
        }
      }
    }
  }

  if !dist.contains_key(&goal) {
    return Ok(Expr::List(vec![].into()));
  }
  // Reconstruct the path goal → start, then reverse.
  let mut path_keys = vec![goal.clone()];
  let mut cur = goal;
  while cur != start {
    match prev.get(&cur) {
      Some(p) => {
        path_keys.push(p.clone());
        cur = p.clone();
      }
      None => return Ok(Expr::List(vec![].into())),
    }
  }
  path_keys.reverse();
  let path: Vec<Expr> = path_keys.iter().map(|k| vexpr[k].clone()).collect();
  Ok(Expr::List(path.into()))
}

/// Total-order wrapper for f64 Dijkstra costs (all non-negative, finite).
mod ordered_f64 {
  #[derive(PartialEq)]
  pub struct OrderedF64(pub f64);
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
    } if gname == "Graph" && gargs.len() >= 2 => {
      match (&gargs[0], &gargs[1]) {
        (Expr::List(v), Expr::List(e)) => (v, e),
        _ => return Ok(unevaluated()),
      }
    }
    _ => return Ok(unevaluated()),
  };
  let vkeys: Vec<String> = vertices.iter().map(expr_to_string).collect();
  let n = vkeys.len();
  let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(edge_exprs.len());
  for e in edge_exprs.iter() {
    match e {
      Expr::FunctionCall {
        name: ename,
        args: eargs,
      } if ename == "UndirectedEdge" && eargs.len() == 2 => {
        let a = vkeys.iter().position(|k| *k == expr_to_string(&eargs[0]));
        let b = vkeys.iter().position(|k| *k == expr_to_string(&eargs[1]));
        match (a, b) {
          (Some(a), Some(b)) => pairs.push((a, b)),
          _ => return Ok(unevaluated()),
        }
      }
      _ => return Ok(unevaluated()),
    }
  }
  let adjacent = |s: usize, t: usize| {
    pairs
      .iter()
      .any(|&(a, b)| (a == s && b == t) || (a == t && b == s))
  };

  if args.len() == 3 {
    let find =
      |e: &Expr| vkeys.iter().position(|k| *k == expr_to_string(e));
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
