//! `ExampleData[{"NetworkGraph", …}]`.
//!
//! A handful of classic networks are real: they are assembled from the
//! publications they come from (`resources/network_graphs.txt.gz`) and carry
//! the names Wolfram's catalogue lists them under, so a script that asks for
//! one of them by name gets the same network from either engine. Every other
//! catalogued network is generated (see [`generated`]) with the vertex and
//! edge counts, directedness and degree statistics of the original.

use std::sync::LazyLock;

use super::{Collection, Entry};
use crate::functions::*;

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

  let compressed = include_bytes!("../../../resources/network_graphs.txt.gz");
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
              Err(_) => str_expr(v),
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

/// The catalogue name `name` stands for, resolving the alternative
/// spellings above.
fn resolve_network_graph_name(name: &str) -> &str {
  NETWORK_GRAPH_ALIASES
    .iter()
    .find(|(alias, _)| *alias == name)
    .map_or(name, |(_, catalogue)| *catalogue)
}

/// The dataset named by `name`, if it is bundled.
fn network_graph(name: &str) -> Option<&'static NetworkGraph> {
  let name = resolve_network_graph_name(name);
  NETWORK_GRAPHS.iter().find(|g| g.name == name)
}

/// The `Graph[…]` for a bundled network.
fn network_graph_expr(g: &NetworkGraph) -> Expr {
  let edges: Vec<Expr> = g
    .edges
    .iter()
    .map(|&(a, b)| {
      call(
        "UndirectedEdge",
        vec![g.vertices[a].clone(), g.vertices[b].clone()],
      )
    })
    .collect();
  call(
    "Graph",
    vec![
      Expr::List(g.vertices.clone().into()),
      Expr::List(edges.into()),
    ],
  )
}

pub struct NetworkGraphs;

impl Collection for NetworkGraphs {
  fn resolve<'n>(&self, name: &'n str) -> &'n str {
    resolve_network_graph_name(name)
  }

  fn value(&self, entry: &Entry, property: Option<&str>) -> Option<Expr> {
    let real = network_graph(entry.name);
    let graph = || match real {
      Some(g) => Some(network_graph_expr(g)),
      None => super::images::cached_opt(
        format!("NetworkGraph/{}", entry.name),
        || Some(generated(entry)),
      ),
    };
    Some(match property {
      None | Some("Graph" | "FullGraph") => graph()?,
      Some("Description") => match real {
        Some(g) => str_expr(&g.description),
        None => str_expr(&describe(entry)),
      },
      Some("LongDescription") => str_expr(&long_description(entry)),
      Some("Source") => match real {
        Some(g) => str_expr(&g.source),
        None => str_expr(super::GENERATED_SOURCE),
      },
      _ => return None,
    })
  }
}

/// Whether the network is directed, from the metadata.
fn directed(entry: &Entry) -> bool {
  matches!(entry.stat("@Graph"), Some(Expr::List(ref f))
    if matches!(f.first(), Some(Expr::Identifier(d)) if d == "True"))
}

fn count(entry: &Entry, column: &str) -> i128 {
  match entry.exact(column) {
    Some(Expr::Integer(n)) => n,
    _ => 0,
  }
}

/// A one-line description in Woxi's words.
fn describe(entry: &Entry) -> String {
  format!(
    "{} network with {} vertices and {} edges",
    if directed(entry) {
      "A directed"
    } else {
      "An undirected"
    },
    count(entry, "VertexCount"),
    count(entry, "EdgeCount")
  )
}

fn long_description(entry: &Entry) -> String {
  let name = match entry.exact("Name") {
    Some(Expr::String(ref n)) => n.clone(),
    _ => entry.name.to_string(),
  };
  if network_graph(entry.name).is_some() {
    format!(
      "{}. The network is the published one ({}).",
      describe(entry),
      name
    )
  } else {
    format!(
      "{}. Woxi generates this network as a stand-in for the {} dataset: \
       it has the same size and direction, but its edges are random.",
      describe(entry),
      name
    )
  }
}

/// A generated network with the recorded vertex and edge counts,
/// direction, degree range, vertex kind and (if any) edge weights.
fn generated(entry: &Entry) -> Expr {
  use super::graph_models::{GraphSpec, generate};
  let n = count(entry, "VertexCount") as usize;
  let m = count(entry, "EdgeCount") as usize;
  let stats = match entry.stat("@Graph") {
    Some(Expr::List(ref f)) => f.iter().cloned().collect::<Vec<_>>(),
    _ => Vec::new(),
  };
  let flag =
    |i: usize| matches!(stats.get(i), Some(Expr::Identifier(t)) if t == "True");
  let degrees = super::images::numbers(
    stats
      .get(5)
      .unwrap_or(&Expr::List(crate::ExprList::default())),
  );
  let spec = GraphSpec {
    vertices: n,
    edges: m,
    directed: flag(0),
    multigraph: flag(2),
    loops: !flag(3),
    min_degree: degrees.first().copied().unwrap_or(1.0),
    max_degree: degrees
      .get(1)
      .copied()
      .unwrap_or((2 * m / n.max(1)) as f64 * 4.0),
  };
  let mut rng = entry.rng("graph");
  let edges = generate(&mut rng, &spec);
  // Vertex names: words when the original names its vertices.
  let named = matches!(stats.get(6), Some(Expr::List(h)) if h.iter().any(|x| matches!(x, Expr::Identifier(s) if s == "String")));
  let vertices: Vec<Expr> = if named {
    let mut seen = std::collections::HashSet::new();
    let lex = super::words::Lexicon::English;
    (0..n)
      .map(|i| {
        let mut w = super::words::capitalize(&lex.word(&mut rng));
        if !seen.insert(w.clone()) {
          w = format!("{w}{i}");
          seen.insert(w.clone());
        }
        str_expr(&w)
      })
      .collect()
  } else {
    (1..=n).map(|i| Expr::Integer(i as i128)).collect()
  };
  let head = if spec.directed {
    "DirectedEdge"
  } else {
    "UndirectedEdge"
  };
  let edge_list: Vec<Expr> = edges
    .iter()
    .map(|&(a, b)| call(head, vec![vertices[a].clone(), vertices[b].clone()]))
    .collect();
  let mut args =
    vec![Expr::List(vertices.into()), Expr::List(edge_list.into())];
  if flag(1) {
    let w = stats.get(7).and_then(super::tabular::parse_column_numeric);
    let w = w.unwrap_or_default();
    let mut draws = super::tabular::sorted_draws(&mut rng, &w, edges.len());
    rng.shuffle(&mut draws);
    args.push(rule_expr(
      id_expr("EdgeWeight"),
      Expr::List(
        draws
          .iter()
          .map(|v| super::tabular::quantize(*v, &w))
          .collect::<Vec<_>>()
          .into(),
      ),
    ));
  }
  // Built in final form: the vertices and edges are already evaluated, so
  // the (for large networks, costly) re-evaluation of every edge is skipped.
  call("Graph", args)
}
