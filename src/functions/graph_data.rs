//! Minimal built-in named-graph knowledge base.
//!
//! Woxi has no connection to the Wolfram Knowledgebase's graph atlas (which
//! enumerates thousands of graphs), but a self-contained, deterministic set
//! of common named and parametrized graphs is easy to bundle. This backs
//! `GraphData[]` (every known name), `GraphData[name]` (the `Graph[...]`
//! object), and `GraphData[name, "property"]`.
//!
//! Names follow Wolfram's usual convention (`"PetersenGraph"`,
//! `"CompleteGraphK5"`, `"CycleGraphC6"`, …) but this is not meant to match
//! the Wolfram Knowledgebase's full enumeration to the last entry — it gives
//! Woxi a reproducible graph dataset covering the common named graphs and
//! small parametrized families (complete, cycle, path, star, wheel,
//! complete bipartite).

#[allow(unused_imports)]
use super::*;

struct NamedGraph {
  name: String,
  n: usize,
  /// 1-based vertex labels.
  edges: Vec<(usize, usize)>,
}

fn special_graphs() -> Vec<NamedGraph> {
  let mut octahedral_edges = Vec::new();
  for i in 1..=6 {
    for j in (i + 1)..=6 {
      let antipodal = (i, j) == (1, 2) || (i, j) == (3, 4) || (i, j) == (5, 6);
      if !antipodal {
        octahedral_edges.push((i, j));
      }
    }
  }
  vec![
    NamedGraph {
      name: "PetersenGraph".to_string(),
      n: 10,
      edges: vec![
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 1),
        (6, 8),
        (8, 10),
        (10, 7),
        (7, 9),
        (9, 6),
        (1, 6),
        (2, 7),
        (3, 8),
        (4, 9),
        (5, 10),
      ],
    },
    NamedGraph {
      name: "BullGraph".to_string(),
      n: 5,
      edges: vec![(1, 2), (2, 3), (1, 3), (1, 4), (2, 5)],
    },
    NamedGraph {
      name: "DiamondGraph".to_string(),
      n: 4,
      edges: vec![(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)],
    },
    NamedGraph {
      name: "HouseGraph".to_string(),
      n: 5,
      edges: vec![(1, 2), (2, 3), (3, 4), (4, 1), (1, 5), (2, 5)],
    },
    NamedGraph {
      name: "ButterflyGraph".to_string(),
      n: 5,
      edges: vec![(1, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 5)],
    },
    NamedGraph {
      name: "TetrahedralGraph".to_string(),
      n: 4,
      edges: vec![(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
    },
    NamedGraph {
      name: "CubicalGraph".to_string(),
      n: 8,
      edges: vec![
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 5),
        (1, 5),
        (2, 6),
        (3, 7),
        (4, 8),
      ],
    },
    NamedGraph {
      name: "OctahedralGraph".to_string(),
      n: 6,
      edges: octahedral_edges,
    },
  ]
}

fn complete_graphs() -> Vec<NamedGraph> {
  (3..=7)
    .map(|n| {
      let mut edges = Vec::new();
      for i in 1..=n {
        for j in (i + 1)..=n {
          edges.push((i, j));
        }
      }
      NamedGraph {
        name: format!("CompleteGraphK{n}"),
        n,
        edges,
      }
    })
    .collect()
}

fn cycle_graphs() -> Vec<NamedGraph> {
  (3..=15)
    .map(|n| {
      let mut edges: Vec<(usize, usize)> = (1..n).map(|i| (i, i + 1)).collect();
      edges.push((n, 1));
      NamedGraph {
        name: format!("CycleGraphC{n}"),
        n,
        edges,
      }
    })
    .collect()
}

fn path_graphs() -> Vec<NamedGraph> {
  (2..=15)
    .map(|n| {
      let edges: Vec<(usize, usize)> = (1..n).map(|i| (i, i + 1)).collect();
      NamedGraph {
        name: format!("PathGraphP{n}"),
        n,
        edges,
      }
    })
    .collect()
}

fn star_graphs() -> Vec<NamedGraph> {
  (3..=15)
    .map(|n| {
      let edges: Vec<(usize, usize)> = (2..=n).map(|i| (1, i)).collect();
      NamedGraph {
        name: format!("StarGraphS{n}"),
        n,
        edges,
      }
    })
    .collect()
}

fn wheel_graphs() -> Vec<NamedGraph> {
  // Hub is vertex 1; the rim (vertices 2..=n) forms a cycle, and every rim
  // vertex has a spoke back to the hub.
  (4..=10)
    .map(|n| {
      let mut edges: Vec<(usize, usize)> = (2..n).map(|i| (i, i + 1)).collect();
      edges.push((n, 2));
      edges.extend((2..=n).map(|i| (1, i)));
      NamedGraph {
        name: format!("WheelGraphW{n}"),
        n,
        edges,
      }
    })
    .collect()
}

fn complete_bipartite_graphs() -> Vec<NamedGraph> {
  const SIZES: &[(usize, usize)] = &[
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 6),
    (2, 7),
    (3, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (4, 4),
  ];
  SIZES
    .iter()
    .map(|&(m, k)| {
      let mut edges = Vec::new();
      for i in 1..=m {
        for j in (m + 1)..=(m + k) {
          edges.push((i, j));
        }
      }
      NamedGraph {
        name: format!("CompleteBipartiteGraphK{m}{k}"),
        n: m + k,
        edges,
      }
    })
    .collect()
}

fn all_graphs() -> Vec<NamedGraph> {
  let mut graphs = Vec::new();
  graphs.extend(special_graphs());
  graphs.extend(complete_graphs());
  graphs.extend(cycle_graphs());
  graphs.extend(path_graphs());
  graphs.extend(star_graphs());
  graphs.extend(wheel_graphs());
  graphs.extend(complete_bipartite_graphs());
  graphs
}

fn lookup(name: &str) -> Option<NamedGraph> {
  all_graphs().into_iter().find(|g| g.name == name)
}

fn missing_not_available() -> Expr {
  call1("Missing", Expr::String("NotAvailable".to_string()))
}

/// The `Graph[vertices, edges]` object for a named graph, matching the
/// `Graph[List[...], List[UndirectedEdge[...] ...]]` shape every other
/// graph function (`VertexCount`, `EdgeRules`, `GraphPlot`, …) expects.
fn graph_expr(g: &NamedGraph) -> Expr {
  let verts: Vec<Expr> = (1..=g.n as i128).map(Expr::Integer).collect();
  let edges: Vec<Expr> = g
    .edges
    .iter()
    .map(|&(a, b)| {
      call(
        "UndirectedEdge",
        vec![Expr::Integer(a as i128), Expr::Integer(b as i128)],
      )
    })
    .collect();
  call(
    "Graph",
    vec![Expr::List(verts.into()), Expr::List(edges.into())],
  )
}

/// A property spec may be given as a string (`"VertexCount"`) or, as some
/// older Demonstrations do, a bare symbol (`Image`).
fn property_name(e: &Expr) -> Option<String> {
  match e {
    Expr::String(s) => Some(s.clone()),
    Expr::Identifier(s) => Some(s.clone()),
    _ => None,
  }
}

fn graph_property(
  g: &NamedGraph,
  prop: &str,
) -> Result<Expr, InterpreterError> {
  match prop {
    "VertexCount" | "EdgeCount" | "VertexList" | "EdgeList" | "EdgeRules"
    | "AdjacencyMatrix" => {
      crate::evaluator::evaluate_expr_to_expr(&call1(prop, graph_expr(g)))
    }
    "Image" | "Icon" | "Plot" | "Graph" => Ok(graph_expr(g)),
    "Name" | "StandardName" => Ok(Expr::String(g.name.clone())),
    _ => Ok(missing_not_available()),
  }
}

pub fn graph_data_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match args {
    [] => {
      let names: Vec<Expr> = all_graphs()
        .into_iter()
        .map(|g| Expr::String(g.name))
        .collect();
      Ok(Expr::List(names.into()))
    }
    [Expr::String(name)] => match lookup(name) {
      Some(g) => Ok(graph_expr(&g)),
      None => Ok(missing_not_available()),
    },
    [Expr::String(name), prop] => match lookup(name) {
      Some(g) => match property_name(prop) {
        Some(p) => graph_property(&g, &p),
        None => Ok(unevaluated("GraphData", args)),
      },
      None => Ok(missing_not_available()),
    },
    _ => Ok(unevaluated("GraphData", args)),
  }
}
