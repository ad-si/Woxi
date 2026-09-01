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
//! does. Vertex labellings and hence `"EdgeRules"` follow the
//! Knowledgebase's, so the two agree entry for entry over the covered slice.

#[allow(unused_imports)]
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
}

/// The named entities that are not one of the parametrized families, with
/// Wolfram's vertex labelling and its `"Name"` text.
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
    .collect();
  names.sort_unstable();
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

/// Vertex count and 1-based edge list, in Wolfram's vertex labelling and
/// (sorted) edge order.
fn vertices_and_edges(spec: &GraphSpec) -> (usize, Vec<(usize, usize)>) {
  match *spec {
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
  if let GraphSpec::Special(name) = canon {
    return Expr::String(name.to_string());
  }
  if let Some((name, ..)) = ALIAS_GRAPHS.iter().find(|(_, _, s)| *s == canon) {
    return Expr::String(name.to_string());
  }
  spec_expr(&canon)
}

/// A spec as the `{family, size}` list Wolfram spells it with.
fn spec_expr(spec: &GraphSpec) -> Expr {
  let pair = |family: &str, n: usize| {
    Expr::List(
      vec![Expr::String(family.to_string()), Expr::Integer(n as i128)].into(),
    )
  };
  match *spec {
    GraphSpec::Special(name) => Expr::String(name.to_string()),
    GraphSpec::Complete(n) => pair("Complete", n),
    GraphSpec::Cycle(n) => pair("Cycle", n),
    GraphSpec::Path(n) => pair("Path", n),
    GraphSpec::Star(n) => pair("Star", n),
    GraphSpec::Wheel(n) => pair("Wheel", n),
    GraphSpec::CompleteBipartite(m, k) => Expr::List(
      vec![
        Expr::String("CompleteBipartite".to_string()),
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
    _ => None,
  }
}

pub fn graph_data_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated_call = || Ok(unevaluated("GraphData", args));
  match args {
    [] => Ok(Expr::List(
      all_names()
        .into_iter()
        .map(|n| Expr::String(n.to_string()))
        .collect::<Vec<_>>()
        .into(),
    )),
    // `GraphData["Properties"]` is a property query, not an entity lookup.
    // Wolfram lists 764 properties; Woxi lists the ones it answers.
    [Expr::String(p)] if p == "Properties" => Ok(Expr::List(
      PROPERTIES
        .iter()
        .map(|p| Expr::String((*p).to_string()))
        .collect::<Vec<_>>()
        .into(),
    )),
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
