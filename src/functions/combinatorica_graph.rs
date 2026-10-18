//! The graph-theory core of the legacy `Combinatorica\`` package: graphs built
//! by `FromUnorderedPairs` and the traversal functions that read them.
//!
//! A Combinatorica graph is `Graph[edges, vertices]` (its own `Graph` symbol,
//! not the built-in one): every edge is `{{u, v}}` (a vertex pair plus
//! options) and every vertex `{{x, y}}` (a coordinate pair plus options),
//! so `V[g]` is the length of the vertex list.

use crate::InterpreterError;
use crate::syntax::Expr;
use std::collections::VecDeque;

const GRAPH: &str = "Combinatorica`Graph";

/// The vertex count and (0-based, undirected) edges of a Combinatorica graph.
struct CGraph {
  n: usize,
  edges: Vec<(usize, usize)>,
}

impl CGraph {
  fn parse(expr: &Expr) -> Option<Self> {
    let Expr::FunctionCall { name, args } = expr else {
      return None;
    };
    if name != GRAPH || args.len() != 2 {
      return None;
    }
    let (Expr::List(edges), Expr::List(vertices)) = (&args[0], &args[1]) else {
      return None;
    };
    let n = vertices.len();
    let vertex = |e: &Expr| match e {
      Expr::Integer(i) if *i >= 1 && (*i as usize) <= n => {
        Some(*i as usize - 1)
      }
      _ => None,
    };
    let edges = edges
      .iter()
      .map(|edge| {
        let Expr::List(parts) = edge else { return None };
        let Some(Expr::List(pair)) = parts.first() else {
          return None;
        };
        match pair.as_slice() {
          [u, v] => Some((vertex(u)?, vertex(v)?)),
          _ => None,
        }
      })
      .collect::<Option<Vec<_>>>()?;
    Some(CGraph { n, edges })
  }

  /// Sorted, duplicate-free neighbour lists.
  fn adjacency(&self) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); self.n];
    for &(u, v) in &self.edges {
      adj[u].push(v);
      if u != v {
        adj[v].push(u);
      }
    }
    for list in &mut adj {
      list.sort_unstable();
      list.dedup();
    }
    adj
  }

  /// BFS from `start`: the visiting order, each vertex's distance (`None`
  /// for unreachable vertices) and its parent in the search tree.
  fn bfs(&self, start: usize) -> (Vec<usize>, Vec<Option<usize>>, Vec<usize>) {
    let adj = self.adjacency();
    let mut level = vec![None; self.n];
    let mut parent = vec![usize::MAX; self.n];
    let mut order = vec![start];
    let mut queue = VecDeque::from([start]);
    level[start] = Some(0);
    while let Some(u) = queue.pop_front() {
      for &v in &adj[u] {
        if level[v].is_none() {
          level[v] = level[u].map(|l| l + 1);
          parent[v] = u;
          order.push(v);
          queue.push_back(v);
        }
      }
    }
    (order, level, parent)
  }
}

fn int(i: usize) -> Expr {
  Expr::Integer(i as i128 + 1)
}

fn int_list(items: impl IntoIterator<Item = usize>) -> Expr {
  Expr::List(items.into_iter().map(int).collect())
}

/// The 1-based vertex argument `v` of `g`, if it names one.
fn vertex_arg(g: &CGraph, e: &Expr) -> Option<usize> {
  match e {
    Expr::Integer(i) if *i >= 1 && (*i as usize) <= g.n => {
      Some(*i as usize - 1)
    }
    _ => None,
  }
}

/// `FromUnorderedPairs[{{u, v}, …}]`: the graph on vertices `1..Max[pairs]`
/// with one edge per distinct pair, drawn on a circle.
fn from_unordered_pairs(args: &[Expr]) -> Option<Expr> {
  let [Expr::List(pairs)] = args else {
    return None;
  };
  let mut edges: Vec<(i128, i128)> = Vec::new();
  for pair in pairs {
    let Expr::List(p) = pair else { return None };
    match p.as_slice() {
      [Expr::Integer(a), Expr::Integer(b)] if *a >= 1 && *b >= 1 => {
        edges.push((*a.min(b), *a.max(b)));
      }
      _ => return None,
    }
  }
  edges.sort_unstable();
  edges.dedup();
  let n = edges.iter().map(|&(_, b)| b).max().unwrap_or(0);
  let edge_list = edges
    .into_iter()
    .map(|(a, b)| {
      Expr::List(
        vec![Expr::List(vec![Expr::Integer(a), Expr::Integer(b)].into())]
          .into(),
      )
    })
    .collect::<Vec<_>>();
  let vertex_list = (0..n)
    .map(|k| {
      let t = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
      Expr::List(
        vec![Expr::List(
          vec![Expr::Real(t.cos()), Expr::Real(t.sin())].into(),
        )]
        .into(),
      )
    })
    .collect::<Vec<_>>();
  Some(crate::helpers::call(
    GRAPH,
    vec![Expr::List(edge_list.into()), Expr::List(vertex_list.into())],
  ))
}

/// Dispatches the `Combinatorica\`` graph functions. `None` leaves the call
/// unevaluated (wrong argument shapes, or not a Combinatorica graph).
pub fn combinatorica_graph_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  Ok(
    evaluate(name, args)
      .unwrap_or_else(|| crate::helpers::unevaluated(name, args)),
  )
}

fn evaluate(name: &str, args: &[Expr]) -> Option<Expr> {
  if name == "Combinatorica`FromUnorderedPairs" {
    return from_unordered_pairs(args);
  }
  let g = CGraph::parse(args.first()?)?;
  match (name, args.len()) {
    ("Combinatorica`V", 1) => Some(Expr::Integer(g.n as i128)),
    ("Combinatorica`ToAdjacencyLists", 1) => Some(Expr::List(
      g.adjacency()
        .into_iter()
        .map(int_list)
        .collect::<Vec<_>>()
        .into(),
    )),
    ("Combinatorica`ConnectedComponents", 1) => {
      let mut seen = vec![false; g.n];
      let mut components = Vec::new();
      for start in 0..g.n {
        if seen[start] {
          continue;
        }
        let (mut order, _, _) = g.bfs(start);
        for &v in &order {
          seen[v] = true;
        }
        order.sort_unstable();
        components.push(int_list(order));
      }
      Some(Expr::List(components.into()))
    }
    // The vertices in the order a breadth-first search from `v` visits them
    // (neighbours taken in increasing order); unreachable vertices are
    // left out.
    ("Combinatorica`BreadthFirstTraversal", 2) => {
      let start = vertex_arg(&g, &args[1])?;
      Some(int_list(g.bfs(start).0))
    }
    ("Combinatorica`ShortestPath", 3) => {
      let start = vertex_arg(&g, &args[1])?;
      let end = vertex_arg(&g, &args[2])?;
      let (_, level, parent) = g.bfs(start);
      if level[end].is_none() {
        return Some(Expr::List(vec![].into()));
      }
      let mut path = vec![end];
      while *path.last()? != start {
        path.push(parent[*path.last()?]);
      }
      path.reverse();
      Some(int_list(path))
    }
    _ => None,
  }
}
