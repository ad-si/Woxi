//! `ExampleData` — the bundled example datasets.
//!
//! Wolfram ships a large catalogue of example data grouped by type
//! (`"NetworkGraph"`, `"Text"`, `"Matrix"`, …). Woxi implements the same
//! four call forms against the datasets it bundles:
//!
//! - `ExampleData[]` — the available types.
//! - `ExampleData["type"]` — the available `{"type", "name"}` entries.
//! - `ExampleData[{"type", "name"}]` — the data itself.
//! - `ExampleData[{"type", "name"}, "property"]` — one property of it.
//!
//! Only `"NetworkGraph"` is bundled so far, and only with the classic
//! networks listed in `resources/network_graphs.txt.gz`; a type or name
//! that is not bundled stays unevaluated rather than returning wrong data.
//! The bundled networks are assembled from the publications they come from
//! and carry the names Wolfram's catalogue lists them under, so a script
//! that asks for one of them by name gets the same data from either engine.

use std::sync::LazyLock;

use crate::InterpreterError;
use crate::syntax::{Expr, unevaluated};

/// One bundled network dataset.
struct NetworkGraph {
  name: &'static str,
  description: String,
  source: String,
  /// The vertices in Wolfram's order — integers when the dataset numbers
  /// its nodes, strings when it names them.
  vertices: Vec<Expr>,
  /// Undirected edges as index pairs into `vertices`.
  edges: Vec<(usize, usize)>,
}

/// The bundled `"NetworkGraph"` datasets, in catalogue order.
///
/// The resource is a line-oriented text file: one `:name`, `:description`,
/// `:source`, `:vertices` (a `|`-separated list) and `:edges` (a
/// comma-separated list of `i-j` index pairs) per dataset.
static NETWORK_GRAPHS: LazyLock<Vec<NetworkGraph>> = LazyLock::new(|| {
  use flate2::read::GzDecoder;
  use std::io::Read;

  let compressed = include_bytes!("../../resources/network_graphs.txt.gz");
  let mut decoder = GzDecoder::new(&compressed[..]);
  let mut text = String::new();
  decoder
    .read_to_string(&mut text)
    .expect("failed to decompress the example network graphs");

  let mut out: Vec<NetworkGraph> = Vec::new();
  for line in text.lines() {
    let Some((key, value)) = line.split_once(' ') else {
      continue;
    };
    if key == ":name" {
      out.push(NetworkGraph {
        // Leaked so the name can be handed out as a `&'static str` for the
        // lifetime of the process, like the other bundled data tables.
        name: Box::leak(value.to_string().into_boxed_str()),
        description: String::new(),
        source: String::new(),
        vertices: Vec::new(),
        edges: Vec::new(),
      });
    } else {
      let Some(current) = out.last_mut() else {
        continue;
      };
      match key {
        ":description" => current.description = value.to_string(),
        ":source" => current.source = value.to_string(),
        ":vertices" => {
          current.vertices = value
            .split('|')
            .map(|v| match v.parse::<i128>() {
              Ok(n) => Expr::Integer(n),
              Err(_) => Expr::String(v.to_string()),
            })
            .collect();
        }
        ":edges" => {
          current.edges = value
            .split(',')
            .filter_map(|e| {
              let (a, b) = e.split_once('-')?;
              Some((a.parse().ok()?, b.parse().ok()?))
            })
            .collect();
        }
        _ => {}
      }
    }
  }
  out
});

/// A second spelling Wolfram resolves to a catalogue name, so that a script
/// written against either one runs: `ExampleData[{"NetworkGraph",
/// "ZacharysKarateClub"}]` and `…"ZacharyKarateClub"…` are the same network.
const NETWORK_GRAPH_ALIASES: &[(&str, &str)] = &[
  ("ZacharysKarateClub", "ZacharyKarateClub"),
  ("BooksAboutUSPolitics", "USPoliticsBooks"),
];

/// The dataset named by `name`, if it is bundled.
fn network_graph(name: &str) -> Option<&'static NetworkGraph> {
  let name = NETWORK_GRAPH_ALIASES
    .iter()
    .find(|(alias, _)| *alias == name)
    .map_or(name, |(_, catalogue)| *catalogue);
  NETWORK_GRAPHS.iter().find(|g| g.name == name)
}

/// The properties `ExampleData[{"NetworkGraph", …}, prop]` understands, in
/// the alphabetical order Wolfram reports them in.
const NETWORK_GRAPH_PROPERTIES: &[&str] = &[
  "AdjacencyMatrix",
  "Description",
  "EdgeCount",
  "EdgeRules",
  "Graph",
  "Name",
  "Source",
  "VertexCount",
  "VertexList",
];

/// The `Graph[…]` for a bundled network.
fn network_graph_expr(g: &NetworkGraph) -> Expr {
  let edges: Vec<Expr> = g
    .edges
    .iter()
    .map(|&(a, b)| Expr::FunctionCall {
      name: "UndirectedEdge".to_string(),
      args: vec![g.vertices[a].clone(), g.vertices[b].clone()].into(),
    })
    .collect();
  Expr::FunctionCall {
    name: "Graph".to_string(),
    args: vec![
      Expr::List(g.vertices.clone().into()),
      Expr::List(edges.into()),
    ]
    .into(),
  }
}

/// One property of a bundled network, or `None` when the property is not
/// one this dataset type provides.
fn network_graph_property(g: &NetworkGraph, property: &str) -> Option<Expr> {
  let string = |s: &str| Expr::String(s.to_string());
  Some(match property {
    "Graph" => network_graph_expr(g),
    "Name" => string(g.name),
    "Description" => string(&g.description),
    "Source" => string(&g.source),
    "VertexCount" => Expr::Integer(g.vertices.len() as i128),
    "EdgeCount" => Expr::Integer(g.edges.len() as i128),
    "VertexList" => Expr::List(g.vertices.clone().into()),
    "EdgeRules" => Expr::List(
      g.edges
        .iter()
        .map(|&(a, b)| Expr::Rule {
          pattern: Box::new(g.vertices[a].clone()),
          replacement: Box::new(g.vertices[b].clone()),
        })
        .collect::<Vec<_>>()
        .into(),
    ),
    "AdjacencyMatrix" => {
      let n = g.vertices.len();
      let mut rows = vec![vec![0i128; n]; n];
      for &(a, b) in &g.edges {
        rows[a][b] = 1;
        rows[b][a] = 1;
      }
      Expr::List(
        rows
          .into_iter()
          .map(|row| {
            Expr::List(
              row
                .into_iter()
                .map(Expr::Integer)
                .collect::<Vec<_>>()
                .into(),
            )
          })
          .collect::<Vec<_>>()
          .into(),
      )
    }
    "Properties" => Expr::List(
      NETWORK_GRAPH_PROPERTIES
        .iter()
        .map(|p| string(p))
        .collect::<Vec<_>>()
        .into(),
    ),
    _ => return None,
  })
}

/// The example-data types Woxi bundles.
const TYPES: &[&str] = &["NetworkGraph"];

/// `ExampleData[…]` — see the module documentation for the call forms.
pub fn example_data_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("ExampleData", args));

  // ExampleData[] — the available types.
  if args.is_empty() {
    return Ok(Expr::List(
      TYPES
        .iter()
        .map(|t| Expr::String(t.to_string()))
        .collect::<Vec<_>>()
        .into(),
    ));
  }

  // ExampleData["type"] — the entries of that type, each a {type, name} pair.
  if let Expr::String(kind) = &args[0]
    && args.len() == 1
  {
    if kind != "NetworkGraph" {
      return unevaluated();
    }
    return Ok(Expr::List(
      NETWORK_GRAPHS
        .iter()
        .map(|g| {
          Expr::List(
            vec![Expr::String(kind.clone()), Expr::String(g.name.to_string())]
              .into(),
          )
        })
        .collect::<Vec<_>>()
        .into(),
    ));
  }

  // ExampleData[{"type", "name"}] / ExampleData[{"type", "name"}, "prop"]
  let Expr::List(spec) = &args[0] else {
    return unevaluated();
  };
  let (Some(Expr::String(kind)), Some(Expr::String(name))) =
    (spec.first(), spec.get(1))
  else {
    return unevaluated();
  };
  if kind != "NetworkGraph" || args.len() > 2 {
    return unevaluated();
  }
  let Some(graph) = network_graph(name) else {
    return unevaluated();
  };
  let property = match args.get(1) {
    None => "Graph",
    Some(Expr::String(p)) => p.as_str(),
    Some(_) => return unevaluated(),
  };
  match network_graph_property(graph, property) {
    Some(value) => Ok(value),
    None => unevaluated(),
  }
}
