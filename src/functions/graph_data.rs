//! Minimal built-in named-graph knowledge base.
//!
//! Woxi has no connection to the Wolfram Knowledgebase's graph atlas (which
//! enumerates ~12500 graphs), but a self-contained, deterministic slice of it
//! is easy to bundle. This backs `GraphData[]` (every known name),
//! `GraphData[spec]` (the `Graph[...]` object), and
//! `GraphData[spec, "property"]`.
//!
//! Both of Wolfram's spellings are supported: the *named* entities
//! (`"PetersenGraph"`, `"CubicalGraph"`, …) and the *parametrized* specs
//! (`{"Complete", n}`, `{"Cycle", n}`, `{"Path", n}`, `{"Star", n}`,
//! `{"Wheel", n}`, `{"CompleteBipartite", {m, k}}`). Names Wolfram does not
//! have — Woxi used to invent `"CompleteGraphK4"`, `"CycleGraphC6"` and
//! friends — are *not* accepted: an unknown entity leaves the call
//! unevaluated after a `GraphData::notent` message, exactly as wolframscript
//! does. Vertex labelings and hence `"EdgeRules"` follow the
//! Knowledgebase's, so the two agree entry for entry over the covered slice.

use super::*;

/// A graph this module knows how to build. Named entities that *are* one of
/// the parametrized families (a tetrahedral graph is `K4`) are stored as that
/// family, so the two spellings can never produce different edges.
#[derive(Clone, PartialEq, Eq)]
enum GraphSpec {
  /// A one-off named graph, held as its 1-based edge list.
  Special(&'static str),
  Complete(usize),
  Cycle(usize),
  Path(usize),
  Star(usize),
  Wheel(usize),
  CompleteBipartite(usize, usize),
  /// A ladder of `n` rungs whose ends are joined with a half twist.
  MoebiusLadder(usize),
  /// The n-gonal prism: two n-cycles joined by rungs.
  Prism(usize),
  /// The n-gonal antiprism: two n-cycles joined by a zigzag band.
  Antiprism(usize),
  /// The skeleton of a solid PolyhedronData knows, by its graph name.
  Skeleton(&'static str),
}

include!("graph_data_skeletons.rs");

/// The solid whose skeleton a skeleton graph is (the first listed when
/// several solids share a skeleton).
fn skeleton_solid(graph: &str) -> Option<&'static str> {
  SKELETON_GRAPHS
    .iter()
    .find(|(g, ..)| *g == graph)
    .map(|&(_, solid, _)| solid)
}

/// The named entities that are not one of the parametrized families, with
/// Wolfram's vertex labeling and its `"Name"` text.
static SPECIAL_GRAPHS: &[(&str, &str, usize, &[(usize, usize)])] = &[
  (
    "PetersenGraph",
    "Petersen graph",
    10,
    &[
      (1, 3),
      (1, 4),
      (1, 6),
      (2, 4),
      (2, 5),
      (2, 7),
      (3, 5),
      (3, 8),
      (4, 9),
      (5, 10),
      (6, 7),
      (6, 10),
      (7, 8),
      (8, 9),
      (9, 10),
    ],
  ),
  (
    "BullGraph",
    "bull graph",
    5,
    &[(1, 4), (1, 5), (2, 4), (3, 5), (4, 5)],
  ),
  (
    "DiamondGraph",
    "diamond graph",
    4,
    &[(1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
  ),
  (
    "HouseGraph",
    "house graph",
    5,
    &[(1, 3), (1, 4), (1, 5), (2, 4), (2, 5), (3, 5)],
  ),
  (
    "ButterflyGraph",
    "butterfly graph",
    5,
    &[(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (4, 5)],
  ),
  (
    "CubicalGraph",
    "cubical graph",
    8,
    &[
      (1, 2),
      (1, 3),
      (1, 5),
      (2, 4),
      (2, 6),
      (3, 4),
      (3, 7),
      (4, 8),
      (5, 6),
      (5, 7),
      (6, 8),
      (7, 8),
    ],
  ),
  (
    "OctahedralGraph",
    "octahedral graph",
    6,
    &[
      (1, 2),
      (1, 3),
      (1, 4),
      (1, 5),
      (2, 3),
      (2, 4),
      (2, 6),
      (3, 5),
      (3, 6),
      (4, 5),
      (4, 6),
      (5, 6),
    ],
  ),
  (
    "IcosahedralGraph",
    "icosahedral graph",
    12,
    &[
      (1, 3),
      (1, 5),
      (1, 6),
      (1, 9),
      (1, 10),
      (2, 4),
      (2, 7),
      (2, 8),
      (2, 11),
      (2, 12),
      (3, 7),
      (3, 8),
      (3, 9),
      (3, 10),
      (4, 5),
      (4, 6),
      (4, 11),
      (4, 12),
      (5, 6),
      (5, 9),
      (5, 11),
      (6, 10),
      (6, 12),
      (7, 8),
      (7, 9),
      (7, 11),
      (8, 10),
      (8, 12),
      (9, 11),
      (10, 12),
    ],
  ),
  (
    "DodecahedralGraph",
    "dodecahedral graph",
    20,
    &[
      (1, 14),
      (1, 15),
      (1, 16),
      (2, 5),
      (2, 6),
      (2, 13),
      (3, 7),
      (3, 14),
      (3, 19),
      (4, 8),
      (4, 15),
      (4, 20),
      (5, 11),
      (5, 19),
      (6, 12),
      (6, 20),
      (7, 11),
      (7, 16),
      (8, 12),
      (8, 16),
      (9, 10),
      (9, 14),
      (9, 17),
      (10, 15),
      (10, 18),
      (11, 12),
      (13, 17),
      (13, 18),
      (17, 19),
      (18, 20),
    ],
  ),
];

/// Named entities that are an alias for a parametrized family member. The
/// name is what `"StandardName"` reports for that spec, so the mapping runs
/// both ways: `GraphData["TetrahedralGraph"]` is `GraphData[{"Complete", 4}]`,
/// and `GraphData[{"Complete", 4}, "StandardName"]` is `"TetrahedralGraph"`.
static ALIAS_GRAPHS: &[(&str, &str, GraphSpec)] = &[
  ("SingletonGraph", "singleton graph", GraphSpec::Complete(1)),
  ("TriangleGraph", "triangle graph", GraphSpec::Cycle(3)),
  ("SquareGraph", "square graph", GraphSpec::Cycle(4)),
  ("ClawGraph", "claw graph", GraphSpec::Star(4)),
  (
    "TetrahedralGraph",
    "tetrahedral graph",
    GraphSpec::Complete(4),
  ),
  ("PentatopeGraph", "pentatope graph", GraphSpec::Complete(5)),
  (
    "UtilityGraph",
    "utility graph",
    GraphSpec::CompleteBipartite(3, 3),
  ),
  ("WagnerGraph", "Wagner graph", GraphSpec::MoebiusLadder(4)),
];

/// The names `GraphData[]` enumerates. Wolfram's own list has ~12500 entries
/// including every small graph by its canonical code; Woxi lists the named
/// entities it actually carries. Every name here answers every property, and
/// no name Wolfram lacks appears.
fn all_names() -> Vec<&'static str> {
  let mut names: Vec<&'static str> = SPECIAL_GRAPHS
    .iter()
    .map(|&(name, ..)| name)
    .chain(ALIAS_GRAPHS.iter().map(|(name, ..)| *name))
    .chain(SKELETON_GRAPHS.iter().map(|&(name, ..)| name))
    .collect();
  names.sort_unstable();
  names.dedup();
  names
}

/// Wolfram collapses a parametrized spec onto whichever spelling it considers
/// canonical: `{"Complete", 2}` is a `{"Path", 2}`, `{"Star", 4}` is the claw
/// graph, `{"CompleteBipartite", {1, k}}` is a star. Applied before reading
/// `"Name"`/`"StandardName"`, which is where the choice shows.
fn canonical(spec: &GraphSpec) -> GraphSpec {
  match *spec {
    GraphSpec::Complete(1) | GraphSpec::Path(1) => GraphSpec::Complete(1),
    GraphSpec::Complete(2) => GraphSpec::Path(2),
    GraphSpec::Complete(3) => GraphSpec::Cycle(3),
    GraphSpec::Star(n) if n <= 3 => GraphSpec::Path(n),
    GraphSpec::Wheel(4) => GraphSpec::Complete(4),
    GraphSpec::CompleteBipartite(1, 1) => GraphSpec::Path(2),
    GraphSpec::CompleteBipartite(1, 2) => GraphSpec::Path(3),
    GraphSpec::CompleteBipartite(1, k) => GraphSpec::Star(k + 1),
    GraphSpec::CompleteBipartite(2, 2) => GraphSpec::Cycle(4),
    GraphSpec::MoebiusLadder(3) => GraphSpec::CompleteBipartite(3, 3),
    GraphSpec::Prism(4) => GraphSpec::Special("CubicalGraph"),
    GraphSpec::Antiprism(3) => GraphSpec::Special("OctahedralGraph"),
    ref other => other.clone(),
  }
}

/// The parametrized spec a named entity stands for, if it is an alias.
fn alias_spec(name: &str) -> Option<GraphSpec> {
  ALIAS_GRAPHS
    .iter()
    .find(|(n, ..)| *n == name)
    .map(|(_, _, spec)| spec.clone())
}

/// Resolve a `GraphData` first argument to a spec, or `None` if it is not an
/// entity Woxi knows.
fn resolve(arg: &Expr) -> Option<GraphSpec> {
  match arg {
    Expr::String(name) => {
      if let Some(spec) = alias_spec(name) {
        return Some(spec);
      }
      SPECIAL_GRAPHS
        .iter()
        .find(|&&(n, ..)| n == name)
        .map(|&(n, ..)| GraphSpec::Special(n))
        .or_else(|| {
          SKELETON_GRAPHS
            .iter()
            .find(|&&(n, ..)| n == name)
            .map(|&(n, ..)| GraphSpec::Skeleton(n))
        })
    }
    Expr::List(items) => match items.as_ref() {
      [Expr::String(family), Expr::Integer(n)] if *n >= 1 => {
        let n = *n as usize;
        match family.as_str() {
          "Complete" => Some(GraphSpec::Complete(n)),
          "Cycle" if n >= 3 => Some(GraphSpec::Cycle(n)),
          "Path" => Some(GraphSpec::Path(n)),
          "Star" if n >= 2 => Some(GraphSpec::Star(n)),
          "Wheel" if n >= 4 => Some(GraphSpec::Wheel(n)),
          "MoebiusLadder" if n >= 3 => Some(GraphSpec::MoebiusLadder(n)),
          "Prism" if n >= 3 => Some(GraphSpec::Prism(n)),
          "Antiprism" if n >= 3 => Some(GraphSpec::Antiprism(n)),
          _ => None,
        }
      }
      [Expr::String(family), Expr::List(sizes)]
        if family == "CompleteBipartite" =>
      {
        match sizes.as_ref() {
          [Expr::Integer(m), Expr::Integer(k)] if *m >= 1 && *k >= 1 => {
            Some(GraphSpec::CompleteBipartite(*m as usize, *k as usize))
          }
          _ => None,
        }
      }
      _ => None,
    },
    _ => None,
  }
}

/// Vertex count and 1-based edge list, in Wolfram's vertex labeling and
/// (sorted) edge order.
fn vertices_and_edges(spec: &GraphSpec) -> (usize, Vec<(usize, usize)>) {
  // A family member Wolfram files under a named graph has that graph's
  // labeling (`{"Prism", 4}` is the cubical graph).
  if let canon @ GraphSpec::Special(_) = canonical(spec)
    && canon != *spec
  {
    return vertices_and_edges(&canon);
  }
  let sorted = |n: usize, mut edges: Vec<(usize, usize)>| {
    for e in &mut edges {
      if e.0 > e.1 {
        *e = (e.1, e.0);
      }
    }
    edges.sort_unstable();
    edges.dedup();
    (n, edges)
  };
  match *spec {
    // The 2n-cycle with each vertex joined to the opposite one.
    GraphSpec::MoebiusLadder(n) => {
      let m = 2 * n;
      let mut edges: Vec<(usize, usize)> =
        (1..=m).map(|i| (i, i % m + 1)).collect();
      edges.extend((1..=n).map(|i| (i, i + n)));
      sorted(m, edges)
    }
    GraphSpec::Prism(n) => {
      let mut edges = Vec::new();
      for i in 1..=n {
        edges.push((i, i % n + 1));
        edges.push((n + i, n + i % n + 1));
        edges.push((i, n + i));
      }
      sorted(2 * n, edges)
    }
    GraphSpec::Antiprism(n) => {
      let mut edges = Vec::new();
      for i in 1..=n {
        edges.push((i, i % n + 1));
        edges.push((n + i, n + i % n + 1));
        edges.push((i, n + i));
        edges.push((i % n + 1, n + i));
      }
      sorted(2 * n, edges)
    }
    GraphSpec::Skeleton(name) => {
      let solid = str_expr(skeleton_solid(name).unwrap_or(name));
      let query = |prop: &str| {
        crate::functions::polyhedron_data::polyhedron_data_ast(&[
          solid.clone(),
          str_expr(prop),
        ])
      };
      let n = match query("VertexCount") {
        Ok(Expr::Integer(n)) => n as usize,
        _ => 0,
      };
      let edges: Vec<(usize, usize)> = match &query("EdgeIndices") {
        Ok(Expr::List(items)) => items
          .iter()
          .filter_map(|e| match e {
            Expr::List(ab) => match ab.as_slice() {
              [Expr::Integer(a), Expr::Integer(b)] => {
                Some((*a as usize, *b as usize))
              }
              _ => None,
            },
            _ => None,
          })
          .collect(),
        _ => vec![],
      };
      sorted(n, edges)
    }
    GraphSpec::Special(name) => {
      let &(_, _, n, edges) =
        SPECIAL_GRAPHS.iter().find(|&&(s, ..)| s == name).unwrap();
      (n, edges.to_vec())
    }
    GraphSpec::Complete(n) => {
      let mut edges = Vec::new();
      for i in 1..=n {
        for j in (i + 1)..=n {
          edges.push((i, j));
        }
      }
      (n, edges)
    }
    // Wolfram labels the cycle 1–2–…–n–1 and then sorts the edges, so the
    // wrap-around edge `1–n` lands second rather than last.
    GraphSpec::Cycle(n) => {
      let mut edges: Vec<(usize, usize)> = vec![(1, n)];
      edges.extend((1..n).map(|i| (i, i + 1)));
      edges.sort_unstable();
      (n, edges)
    }
    GraphSpec::Path(n) => (n, (1..n).map(|i| (i, i + 1)).collect()),
    // The hub is the *last* vertex, not the first.
    GraphSpec::Star(n) => (n, (1..n).map(|i| (i, n)).collect()),
    // Rim cycle over 1..n-1 with the hub at n.
    GraphSpec::Wheel(n) => {
      let rim = n - 1;
      let mut edges: Vec<(usize, usize)> = vec![(1, rim)];
      edges.extend((1..rim).map(|i| (i, i + 1)));
      edges.extend((1..=rim).map(|i| (i, n)));
      edges.sort_unstable();
      (n, edges)
    }
    GraphSpec::CompleteBipartite(m, k) => {
      let mut edges = Vec::new();
      for i in 1..=m {
        for j in (m + 1)..=(m + k) {
          edges.push((i, j));
        }
      }
      (m + k, edges)
    }
  }
}

/// `"StandardName"`: the named entity when the canonical form has one, and
/// the spec itself otherwise (`{"Cycle", 6}` has no name of its own).
fn standard_name(spec: &GraphSpec) -> Expr {
  let canon = canonical(spec);
  if let GraphSpec::Special(name) | GraphSpec::Skeleton(name) = canon {
    return str_expr(name);
  }
  if let Some((name, ..)) = ALIAS_GRAPHS.iter().find(|(_, _, s)| *s == canon) {
    return str_expr(name);
  }
  spec_expr(&canon)
}

/// A spec as the `{family, size}` list Wolfram spells it with.
fn spec_expr(spec: &GraphSpec) -> Expr {
  let pair = |family: &str, n: usize| {
    Expr::List(vec![str_expr(family), Expr::Integer(n as i128)].into())
  };
  match *spec {
    GraphSpec::Special(name) => str_expr(name),
    GraphSpec::Complete(n) => pair("Complete", n),
    GraphSpec::Cycle(n) => pair("Cycle", n),
    GraphSpec::Path(n) => pair("Path", n),
    GraphSpec::Star(n) => pair("Star", n),
    GraphSpec::Wheel(n) => pair("Wheel", n),
    GraphSpec::MoebiusLadder(n) => pair("MoebiusLadder", n),
    GraphSpec::Prism(n) => pair("Prism", n),
    GraphSpec::Antiprism(n) => pair("Antiprism", n),
    GraphSpec::Skeleton(name) => str_expr(name),
    GraphSpec::CompleteBipartite(m, k) => Expr::List(
      vec![
        str_expr("CompleteBipartite"),
        Expr::List(
          vec![Expr::Integer(m as i128), Expr::Integer(k as i128)].into(),
        ),
      ]
      .into(),
    ),
  }
}

/// `"Name"`: the human-readable description. Wolfram writes the size prefix
/// with a `‐` (U+2010 HYPHEN), not an ASCII hyphen-minus.
fn readable_name(spec: &GraphSpec) -> String {
  let canon = canonical(spec);
  if let GraphSpec::Special(name) = canon {
    let &(_, readable, ..) =
      SPECIAL_GRAPHS.iter().find(|&&(s, ..)| s == name).unwrap();
    return readable.to_string();
  }
  if let Some((_, readable, _)) =
    ALIAS_GRAPHS.iter().find(|(_, _, s)| *s == canon)
  {
    return (*readable).to_string();
  }
  match canon {
    GraphSpec::Complete(n) => format!("{n}\u{2010}complete graph"),
    GraphSpec::Cycle(n) => format!("{n}\u{2010}cycle graph"),
    GraphSpec::Path(n) => format!("{n}\u{2010}path graph"),
    GraphSpec::Star(n) => format!("{n}\u{2010}star graph"),
    GraphSpec::Wheel(n) => format!("{n}\u{2010}wheel graph"),
    GraphSpec::CompleteBipartite(m, k) => {
      format!("({m},{k})\u{2010}complete bipartite graph")
    }
    GraphSpec::MoebiusLadder(n) => {
      format!("{n}\u{2010}M\u{f6}bius ladder graph")
    }
    GraphSpec::Prism(n) => format!("{n}\u{2010}prism graph"),
    GraphSpec::Antiprism(n) => format!("{n}\u{2010}antiprism graph"),
    // "SmallTriakisOctahedralGraph" is the "small triakis octahedral graph".
    GraphSpec::Skeleton(name) => {
      let mut out = String::new();
      for (i, c) in name.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
          out.push(' ');
        }
        out.extend(c.to_lowercase());
      }
      out
    }
    GraphSpec::Special(_) => unreachable!("handled above"),
  }
}

/// The `Graph[vertices, edges]` object for a spec, matching the
/// `Graph[List[...], List[UndirectedEdge[...] ...]]` shape every other
/// graph function (`VertexCount`, `EdgeRules`, `GraphPlot`, …) expects.
fn graph_expr(spec: &GraphSpec) -> Expr {
  let (n, edge_list) = vertices_and_edges(spec);
  let verts: Vec<Expr> = (1..=n as i128).map(Expr::Integer).collect();
  let edges: Vec<Expr> = edge_list
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

/// The properties `GraphData[entity, property]` answers, sorted — the list
/// `GraphData["Properties"]` returns.
static PROPERTIES: &[&str] = &[
  "AdjacencyMatrix",
  "EdgeCount",
  "EdgeList",
  "EdgeRules",
  "Graph",
  "Name",
  "StandardName",
  "VertexCount",
  "VertexList",
];

fn graph_property(
  spec: &GraphSpec,
  prop: &str,
) -> Option<Result<Expr, InterpreterError>> {
  match prop {
    "VertexCount" | "EdgeCount" | "VertexList" | "EdgeList" | "EdgeRules"
    | "AdjacencyMatrix" => Some(crate::evaluator::evaluate_expr_to_expr(
      &call1(prop, graph_expr(spec)),
    )),
    "Graph" => Some(Ok(graph_expr(spec))),
    "Name" => Some(Ok(Expr::String(readable_name(spec)))),
    "StandardName" => Some(Ok(standard_name(spec))),
    "EdgeIndices" => {
      let (_, edges) = vertices_and_edges(spec);
      Some(Ok(Expr::List(
        edges
          .iter()
          .map(|&(a, b)| {
            Expr::List(
              vec![Expr::Integer(a as i128), Expr::Integer(b as i128)].into(),
            )
          })
          .collect::<Vec<_>>()
          .into(),
      )))
    }
    "Degrees" => Some(eval_on_graph(spec, "VertexDegree")),
    // Woxi's own layout and drawings of the graph (wolframscript's come
    // from its curated embeddings).
    "VertexCoordinates" => Some(eval_on_graph(spec, "GraphEmbedding")),
    "Image" => Some(eval_on_graph(spec, "GraphPlot")),
    "Image3D" => Some(eval_on_graph(spec, "GraphPlot3D")),
    "DegreeSequence" => {
      Some(eval_on_graph(spec, "VertexDegree").and_then(|d| {
        crate::evaluator::evaluate_expr_to_expr(&call("ReverseSort", vec![d]))
      }))
    }
    "Regular" => Some(eval_on_graph(spec, "VertexDegree").map(|d| {
      let regular = match &d {
        Expr::List(items) => {
          let degrees: Vec<String> = items
            .iter()
            .map(crate::syntax::expr_to_input_form)
            .collect();
          degrees.windows(2).all(|w| w[0] == w[1])
        }
        _ => false,
      };
      Expr::Identifier(if regular { "True" } else { "False" }.into())
    })),
    _ => class_predicate(prop).map(|q| eval_on_graph(spec, q)),
  }
}

/// The graph-theoretic test behind a Boolean property / class name.
fn class_predicate(class: &str) -> Option<&'static str> {
  Some(match class {
    "Connected" => "ConnectedGraphQ",
    "Planar" => "PlanarGraphQ",
    "Bipartite" => "BipartiteGraphQ",
    "Eulerian" => "EulerianGraphQ",
    "Tree" => "TreeGraphQ",
    _ => return None,
  })
}

fn eval_on_graph(
  spec: &GraphSpec,
  function: &str,
) -> Result<Expr, InterpreterError> {
  crate::evaluator::evaluate_expr_to_expr(&call1(function, graph_expr(spec)))
}

/// `GraphData[class]`: the known graphs in a class. "Polyhedral" is the
/// skeletons of convex solids; the structural classes are decided by
/// testing every known graph.
fn class_members(class: &str) -> Option<Vec<&'static str>> {
  let names = all_names();
  let test = |name: &str, prop: &str| {
    resolve(&str_expr(name))
      .and_then(|spec| graph_property(&spec, prop))
      .and_then(Result::ok)
      .is_some_and(|v| matches!(v, Expr::Identifier(ref t) if t == "True"))
  };
  Some(match class {
    "Polyhedral" => names
      .into_iter()
      .filter(|n| SKELETON_GRAPHS.iter().any(|&(g, _, p)| g == *n && p))
      .collect(),
    "Nonplanar" => names.into_iter().filter(|n| !test(n, "Planar")).collect(),
    "Regular" => names.into_iter().filter(|n| test(n, "Regular")).collect(),
    _ => {
      class_predicate(class)?;
      names.into_iter().filter(|n| test(n, class)).collect()
    }
  })
}

pub fn graph_data_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated_call = || Ok(unevaluated("GraphData", args));
  match args {
    [] => Ok(Expr::List(all_names().into_iter().map(str_expr).collect())),
    // `GraphData["Properties"]` is a property query, not an entity lookup.
    // Wolfram lists 764 properties; Woxi lists the ones it answers.
    [Expr::String(p)] if p == "Properties" => {
      Ok(Expr::List(PROPERTIES.iter().map(|p| str_expr(p)).collect()))
    }
    [Expr::String(class)] if resolve(&args[0]).is_none() => {
      if let Some(members) = class_members(class) {
        Ok(Expr::List(members.into_iter().map(str_expr).collect()))
      } else {
        crate::emit_message(&format!(
          "GraphData::notent: {class} is not a known entity, class or tag for GraphData. Use GraphData[] for a list of entities."
        ));
        unevaluated_call()
      }
    }
    // A class with a property maps the property over the members.
    [Expr::String(class), prop @ Expr::String(_)]
      if resolve(&args[0]).is_none() && class_members(class).is_some() =>
    {
      let members = class_members(class).unwrap();
      let values = members
        .into_iter()
        .map(|m| graph_data_ast(&[str_expr(m), prop.clone()]))
        .collect::<Result<Vec<_>, _>>()?;
      Ok(Expr::List(values.into()))
    }
    [entity] | [entity, _] => {
      let Some(spec) = resolve(entity) else {
        crate::emit_message(&format!(
          "GraphData::notent: {} is not a known entity, class or tag for GraphData. Use GraphData[] for a list of entities.",
          match entity {
            Expr::String(s) => s.clone(),
            other => crate::syntax::expr_to_input_form(other),
          }
        ));
        return unevaluated_call();
      };
      let Some(property) = args.get(1) else {
        return Ok(graph_expr(&spec));
      };
      // Wolfram only accepts a *string* property: a bare symbol
      // (`GraphData[name, Image]`) is a `notprop` error, not a shorthand.
      let answer = match property {
        Expr::String(prop) => graph_property(&spec, prop),
        _ => None,
      };
      if let Some(result) = answer {
        result
      } else {
        crate::emit_message(&format!(
          "GraphData::notprop: {} is not a known property or size specification for GraphData. Use GraphData[\"Properties\"] for a list of properties.",
          crate::syntax::expr_to_input_form(property)
        ));
        unevaluated_call()
      }
    }
    _ => unevaluated_call(),
  }
}
