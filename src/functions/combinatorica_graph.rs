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
    if name != GRAPH || args.len() < 2 {
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
  Some(circular_graph(n, edges))
}

/// The Combinatorica graph on vertices `1..n`, drawn on a circle, with the
/// given (1-based, ordered) edges.
fn circular_graph(n: i128, edges: Vec<(i128, i128)>) -> Expr {
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
  crate::helpers::call(
    GRAPH,
    vec![Expr::List(edge_list.into()), Expr::List(vertex_list.into())],
  )
}

/// `RandomGraph[n, p]`: each of the `n (n - 1)/2` vertex pairs is an edge
/// with probability `p`.
fn random_graph(args: &[Expr]) -> Option<Expr> {
  let [Expr::Integer(n), p] = args else {
    return None;
  };
  let p = crate::functions::math_ast::try_eval_to_f64(p)?;
  if *n < 0 || *n > 2000 || !(0.0..=1.0).contains(&p) {
    return None;
  }
  let mut edges = Vec::new();
  crate::with_rng(|rng| {
    for a in 1..=*n {
      for b in a + 1..=*n {
        let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        if u < p {
          edges.push((a, b));
        }
      }
    }
  });
  Some(circular_graph(*n, edges))
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
  if name == "Combinatorica`RandomGraph" {
    return random_graph(args);
  }
  if name == "Combinatorica`ShowGraph" {
    return show_graph(args, false);
  }
  if name == "Combinatorica`ShowLabeledGraph" {
    return show_graph(args, true);
  }
  let g = CGraph::parse(args.first()?)?;
  match (name, args.len()) {
    ("Combinatorica`V", 1) => Some(Expr::Integer(g.n as i128)),
    ("Combinatorica`M", 1) => Some(Expr::Integer(g.edges.len() as i128)),
    ("Combinatorica`Edges", 1) => Some(Expr::List(
      g.edges
        .iter()
        .map(|&(a, b)| int_list([a, b]))
        .collect::<Vec<_>>()
        .into(),
    )),
    ("Combinatorica`ToAdjacencyMatrix", 1) => {
      let mut m = vec![vec![0i128; g.n]; g.n];
      for &(a, b) in &g.edges {
        m[a][b] += 1;
        if a != b {
          m[b][a] += 1;
        }
      }
      Some(Expr::List(
        m.into_iter()
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
      ))
    }
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

fn symbol_is(e: &Expr, name: &str) -> bool {
  matches!(e, Expr::Identifier(n) if n == name || n.ends_with(&format!("`{name}")))
}

/// The value of the option `name` (in any context) among `opts`.
fn option<'a>(opts: &'a [Expr], name: &str) -> Option<&'a Expr> {
  opts.iter().find_map(|o| {
    crate::evaluator::dispatch::list_operations::as_rule(o)
      .filter(|(k, _)| symbol_is(k, name))
      .map(|(_, v)| v)
  })
}

fn real_pair(e: &Expr) -> Option<(f64, f64)> {
  let Expr::List(items) = e else { return None };
  match items.as_slice() {
    [x, y] => Some((
      crate::functions::math_ast::try_eval_to_f64(x)?,
      crate::functions::math_ast::try_eval_to_f64(y)?,
    )),
    _ => None,
  }
}

fn pt(x: f64, y: f64) -> Expr {
  Expr::List(vec![Expr::Real(x), Expr::Real(y)].into())
}

/// `ShowGraph[g, opts]` / `ShowLabeledGraph[g, opts]`: the graph drawn as
/// `Graphics`: edges as lines, vertices as disks, and (for the labeled
/// form) the vertex numbers. Per-edge `EdgeColor`, per-vertex `VertexColor`
/// and the graph's `EdgeStyle`, `VertexStyle` and `VertexNumberPosition`
/// options are honoured; the remaining options go to `Graphics`.
fn show_graph(args: &[Expr], labeled: bool) -> Option<Expr> {
  let (graph, extra) = args.split_first()?;
  let Expr::FunctionCall { name, args: gargs } = graph else {
    return None;
  };
  if name != GRAPH || gargs.len() < 2 {
    return None;
  }
  let (Expr::List(edges), Expr::List(vertices)) = (&gargs[0], &gargs[1]) else {
    return None;
  };
  let gopts = &gargs[2..];
  let coords = vertices
    .iter()
    .map(|v| {
      let Expr::List(parts) = v else { return None };
      real_pair(parts.first()?)
    })
    .collect::<Option<Vec<_>>>()?;
  let (mut x0, mut x1, mut y0, mut y1) =
    (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
  for &(x, y) in &coords {
    x0 = x0.min(x);
    x1 = x1.max(x);
    y0 = y0.min(y);
    y1 = y1.max(y);
  }
  if coords.is_empty() {
    (x0, x1, y0, y1) = (0.0, 1.0, 0.0, 1.0);
  }
  let span = (x1 - x0).max(y1 - y0).max(1e-9);
  let pad = 0.1 * span;

  let mut prims: Vec<Expr> = Vec::new();
  let thin = option(gopts, "EdgeStyle")
    .is_some_and(|s| matches!(s, Expr::Identifier(n) if n == "Thin"));
  if thin {
    prims.push(Expr::Identifier("Thin".to_string()));
  }
  for edge in edges {
    let Expr::List(parts) = edge else { return None };
    let Some(Expr::List(pair)) = parts.first() else {
      return None;
    };
    let [Expr::Integer(u), Expr::Integer(v)] = pair.as_slice() else {
      return None;
    };
    let (&(ux, uy), &(vx, vy)) = (
      coords.get((*u as usize).checked_sub(1)?)?,
      coords.get((*v as usize).checked_sub(1)?)?,
    );
    let line = crate::helpers::call(
      "Line",
      vec![Expr::List(vec![pt(ux, uy), pt(vx, vy)].into())],
    );
    match option(&parts[1..], "EdgeColor") {
      Some(color) => prims.push(Expr::List(vec![color.clone(), line].into())),
      None => prims.push(line),
    }
  }

  let radius = match option(gopts, "VertexStyle") {
    Some(Expr::FunctionCall { name, args }) if name == "Disk" => {
      match args.as_slice() {
        [r] => crate::functions::math_ast::try_eval_to_f64(r)? * span,
        _ => 0.03 * span,
      }
    }
    _ => 0.03 * span,
  };
  let (dx, dy) = option(gopts, "VertexNumberPosition")
    .and_then(real_pair)
    .unwrap_or((0.0, 0.0));
  for (i, (v, &(x, y))) in vertices.iter().zip(&coords).enumerate() {
    let Expr::List(parts) = v else { return None };
    let disk = crate::helpers::call("Disk", vec![pt(x, y), Expr::Real(radius)]);
    match option(&parts[1..], "VertexColor") {
      Some(color) => prims.push(Expr::List(vec![color.clone(), disk].into())),
      None => prims.push(disk),
    }
    if labeled {
      prims.push(crate::helpers::call(
        "Text",
        vec![
          Expr::Integer(i as i128 + 1),
          pt(x + dx * span, y + dy * span + 2.0 * radius),
        ],
      ));
    }
  }

  let rule = |k: &str, v: Expr| Expr::Rule {
    pattern: Box::new(Expr::Identifier(k.to_string())),
    replacement: Box::new(v),
  };
  let mut out = vec![Expr::List(prims.into())];
  out.push(rule(
    "AspectRatio",
    Expr::Identifier("Automatic".to_string()),
  ));
  out.push(rule(
    "PlotRange",
    Expr::List(
      vec![
        Expr::List(vec![Expr::Real(x0 - pad), Expr::Real(x1 + pad)].into()),
        Expr::List(vec![Expr::Real(y0 - pad), Expr::Real(y1 + pad)].into()),
      ]
      .into(),
    ),
  ));
  out.extend(extra.iter().cloned());
  Some(crate::helpers::call("Graphics", out))
}
