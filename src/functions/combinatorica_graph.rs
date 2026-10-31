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
  /// `EdgeDirection -> True`: each edge runs from its first vertex only.
  directed: bool,
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
    let directed = option(&args[2..], "EdgeDirection")
      .is_some_and(|d| matches!(d, Expr::Identifier(t) if t == "True"));
    Some(CGraph { n, edges, directed })
  }

  /// Sorted, duplicate-free neighbour lists.
  fn adjacency(&self) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); self.n];
    for &(u, v) in &self.edges {
      adj[u].push(v);
      if u != v && !self.directed {
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

/// Whether `a` and `b` are isomorphic as simple undirected graphs.
fn isomorphic(a: &CGraph, b: &CGraph) -> bool {
  if a.n != b.n {
    return false;
  }
  let (adj_a, adj_b) = (a.adjacency(), b.adjacency());
  let matrix = |adj: &[Vec<usize>]| {
    let mut m = vec![vec![false; a.n]; a.n];
    for (u, list) in adj.iter().enumerate() {
      for &v in list {
        m[u][v] = true;
      }
    }
    m
  };
  let (ma, mb) = (matrix(&adj_a), matrix(&adj_b));
  let deg_a: Vec<usize> = adj_a.iter().map(Vec::len).collect();
  let deg_b: Vec<usize> = adj_b.iter().map(Vec::len).collect();
  let (mut sa, mut sb) = (deg_a.clone(), deg_b.clone());
  sa.sort_unstable();
  sb.sort_unstable();
  if sa != sb {
    return false;
  }
  // Map the vertices of `a`, highest degree first, onto unused vertices of
  // `b` with the same degree that agree on all earlier adjacencies.
  let mut order: Vec<usize> = (0..a.n).collect();
  order.sort_by_key(|&v| std::cmp::Reverse(deg_a[v]));
  fn extend(
    k: usize,
    order: &[usize],
    deg: (&[usize], &[usize]),
    m: (&[Vec<bool>], &[Vec<bool>]),
    image: &mut Vec<usize>,
    used: &mut Vec<bool>,
  ) -> bool {
    if k == order.len() {
      return true;
    }
    let u = order[k];
    for w in 0..used.len() {
      if used[w]
        || deg.0[u] != deg.1[w]
        || (0..k).any(|j| m.0[u][order[j]] != m.1[w][image[order[j]]])
        || m.0[u][u] != m.1[w][w]
      {
        continue;
      }
      image[u] = w;
      used[w] = true;
      if extend(k + 1, order, deg, m, image, used) {
        return true;
      }
      used[w] = false;
    }
    false
  }
  extend(
    0,
    &order,
    (&deg_a, &deg_b),
    (&ma, &mb),
    &mut vec![0; a.n],
    &mut vec![false; a.n],
  )
}

/// The vertex coordinates of a Combinatorica graph.
fn coordinates(expr: &Expr) -> Option<Vec<(f64, f64)>> {
  let Expr::FunctionCall { args, .. } = expr else {
    return None;
  };
  let Expr::List(vertices) = args.get(1)? else {
    return None;
  };
  vertices
    .iter()
    .map(|v| {
      let Expr::List(parts) = v else { return None };
      real_pair(parts.first()?)
    })
    .collect()
}

/// `SpringEmbedding[g]`: a force-directed layout (edges attract, all vertex
/// pairs repel) started from the current coordinates and scaled to the
/// unit square.
fn spring_embedding(g: &CGraph, graph: &Expr) -> Option<Expr> {
  let Expr::FunctionCall { args, .. } = graph else {
    return None;
  };
  let mut pos = coordinates(graph)?;
  let n = g.n;
  if n == 0 {
    return Some(graph.clone());
  }
  // Coincident starting points would exert no direction; spread them out.
  if pos.windows(2).all(|w| w[0] == w[1]) {
    for (k, p) in pos.iter_mut().enumerate() {
      let t = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
      *p = (t.cos(), t.sin());
    }
  }
  let k = (1.0 / n as f64).sqrt();
  let iterations = 200;
  for it in 0..iterations {
    let temp = 0.1 * (1.0 - it as f64 / iterations as f64);
    let mut disp = vec![(0.0f64, 0.0f64); n];
    for u in 0..n {
      for v in (u + 1)..n {
        let (dx, dy) = (pos[u].0 - pos[v].0, pos[u].1 - pos[v].1);
        let d = (dx * dx + dy * dy).sqrt().max(1e-9);
        let f = k * k / d;
        disp[u].0 += dx / d * f;
        disp[u].1 += dy / d * f;
        disp[v].0 -= dx / d * f;
        disp[v].1 -= dy / d * f;
      }
    }
    for &(u, v) in &g.edges {
      if u == v {
        continue;
      }
      let (dx, dy) = (pos[u].0 - pos[v].0, pos[u].1 - pos[v].1);
      let d = (dx * dx + dy * dy).sqrt().max(1e-9);
      let f = d * d / k;
      disp[u].0 -= dx / d * f;
      disp[u].1 -= dy / d * f;
      disp[v].0 += dx / d * f;
      disp[v].1 += dy / d * f;
    }
    for u in 0..n {
      let (dx, dy) = disp[u];
      let d = (dx * dx + dy * dy).sqrt().max(1e-9);
      let step = d.min(temp);
      pos[u].0 += dx / d * step;
      pos[u].1 += dy / d * step;
    }
  }
  let span = |f: fn(&(f64, f64)) -> f64| {
    let lo = pos.iter().map(f).fold(f64::INFINITY, f64::min);
    let hi = pos.iter().map(f).fold(f64::NEG_INFINITY, f64::max);
    (lo, if hi > lo { hi - lo } else { 1.0 })
  };
  let ((x0, wx), (y0, wy)) = (span(|p| p.0), span(|p| p.1));
  let vertices = pos
    .iter()
    .map(|&(x, y)| Expr::List(vec![pt((x - x0) / wx, (y - y0) / wy)].into()))
    .collect::<Vec<_>>();
  Some(crate::helpers::call(
    GRAPH,
    std::iter::once(args[0].clone())
      .chain(std::iter::once(Expr::List(vertices.into())))
      .chain(args[2..].iter().cloned())
      .collect(),
  ))
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
  let n = edges.iter().map(|&(_, b)| b).max().unwrap_or(0);
  Some(circular_graph(n, edges, false))
}

/// `FromOrderedPairs[pairs]`: the directed graph with one edge per pair, in
/// the order given, on the vertices up to the largest one named.
fn from_ordered_pairs(args: &[Expr]) -> Option<Expr> {
  let [Expr::List(pairs)] = args else {
    return None;
  };
  let mut edges: Vec<(i128, i128)> = Vec::new();
  for pair in pairs {
    let Expr::List(p) = pair else { return None };
    match p.as_slice() {
      [Expr::Integer(a), Expr::Integer(b)] if *a >= 1 && *b >= 1 => {
        edges.push((*a, *b));
      }
      _ => return None,
    }
  }
  let n = edges.iter().map(|&(a, b)| a.max(b)).max().unwrap_or(0);
  Some(circular_graph(n, edges, true))
}

/// `DeleteEdges[g, {u, v}]` / `DeleteEdges[g, {{u1, v1}, …}]`: `g` with one
/// edge per named pair removed, scanning the edges in order (either
/// orientation names an edge of an undirected graph); with a trailing `All`
/// every edge joining a named pair goes. Pairs naming no edge are ignored.
fn delete_edges(args: &[Expr]) -> Option<Expr> {
  let (graph, spec, all) = match args {
    [g, s] => (g, s, false),
    [g, s, Expr::Identifier(flag)] if flag == "All" => (g, s, true),
    _ => return None,
  };
  let g = CGraph::parse(graph)?;
  let Expr::List(items) = spec else {
    return None;
  };
  let pair = |e: &Expr| match e {
    Expr::List(p) => match p.as_slice() {
      [Expr::Integer(a), Expr::Integer(b)] => Some((*a, *b)),
      _ => None,
    },
    _ => None,
  };
  let mut wanted: Vec<Option<(i128, i128)>> = match pair(spec) {
    Some(p) => vec![Some(p)],
    None => items
      .iter()
      .map(|e| pair(e).map(Some))
      .collect::<Option<_>>()?,
  };
  if !g.directed {
    let reversed: Vec<_> =
      wanted.iter().map(|p| p.map(|(a, b)| (b, a))).collect();
    wanted.extend(reversed);
  }
  let keep: Vec<bool> = g
    .edges
    .iter()
    .map(|&(u, v)| {
      let edge = (u as i128 + 1, v as i128 + 1);
      match wanted.iter().position(|p| *p == Some(edge)) {
        Some(i) => {
          if !all {
            wanted[i] = None;
          }
          false
        }
        None => true,
      }
    })
    .collect();
  let Expr::FunctionCall { name, args: gargs } = graph else {
    return None;
  };
  let Expr::List(edges) = &gargs[0] else {
    return None;
  };
  let mut parts = gargs.to_vec();
  parts[0] = Expr::List(
    edges
      .iter()
      .zip(keep)
      .filter(|(_, k)| *k)
      .map(|(e, _)| e.clone())
      .collect::<Vec<_>>()
      .into(),
  );
  Some(crate::helpers::call(name, parts))
}

/// The Combinatorica graph on vertices `1..n`, drawn on a circle with
/// vertex `k` at angle `2 Pi k/n` (Combinatorica's `CircularEmbedding`,
/// coordinates chopped so `Cos[2 Pi]`'s sine is an exact 0), with the given
/// (1-based, ordered) edges.
fn circular_graph(n: i128, edges: Vec<(i128, i128)>, directed: bool) -> Expr {
  let chop = |x: f64| {
    if x.abs() < 1e-10 {
      Expr::Integer(0)
    } else {
      Expr::Real(x)
    }
  };
  let edge_list = edges
    .into_iter()
    .map(|(a, b)| {
      Expr::List(
        vec![Expr::List(vec![Expr::Integer(a), Expr::Integer(b)].into())]
          .into(),
      )
    })
    .collect::<Vec<_>>();
  let vertex_list = (1..=n)
    .map(|k| {
      let t = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
      Expr::List(
        vec![Expr::List(vec![chop(t.cos()), chop(t.sin())].into())].into(),
      )
    })
    .collect::<Vec<_>>();
  let mut parts =
    vec![Expr::List(edge_list.into()), Expr::List(vertex_list.into())];
  if directed {
    parts.push(Expr::Rule {
      pattern: Box::new(Expr::Identifier("EdgeDirection".to_string())),
      replacement: Box::new(Expr::Identifier("True".to_string())),
    });
  }
  crate::helpers::call(GRAPH, parts)
}

/// `SetGraphOptions[g, spec]` / `SetGraphOptions[g, {spec1, spec2, …}]`
/// with each spec `{v1, v2, …, opts}` (vertices) or `{{u1, v1}, …, opts}`
/// (edges), then an optional `All` / `One` flag and graph options. A vertex
/// or edge takes the options of the first spec naming it (with `One` each
/// named edge occurrence is used up by the first edge it matches); the
/// pairs of an undirected graph and of its edge specs are sorted first.
/// Options merge as Combinatorica's `Merge` does: a new value replaces the
/// old one in place, new names are appended.
fn set_graph_options(args: &[Expr]) -> Option<Expr> {
  let (graph, rest) = args.split_first()?;
  let Expr::FunctionCall { name, args: gargs } = graph else {
    return None;
  };
  if name != GRAPH || gargs.len() < 2 {
    return None;
  }
  let as_rule = crate::evaluator::dispatch::list_operations::as_rule;
  let is_rule = |e: &Expr| as_rule(e).is_some();
  let (spec, rest) = match rest.split_first() {
    Some((Expr::List(l), rest)) => (l.to_vec(), rest),
    _ => (Vec::new(), rest),
  };
  let (one, opts) = match rest.split_first() {
    Some((Expr::Identifier(f), rest)) if f == "One" => (true, rest),
    Some((Expr::Identifier(_), rest)) => (false, rest),
    _ => (false, rest),
  };
  if !opts.iter().all(is_rule) {
    return None;
  }
  let pair = |e: &Expr| match e {
    Expr::List(p) => match p.as_slice() {
      [Expr::Integer(a), Expr::Integer(b)] => Some((*a, *b)),
      _ => None,
    },
    _ => None,
  };
  // `{targets.., opts}` split into its targets and options, if it has the
  // shape `target..` followed by `opts___` (at least one target).
  let split =
    |items: &[Expr], vertex: bool| -> Option<(Vec<Expr>, Vec<Expr>)> {
      let k = items.iter().position(is_rule).unwrap_or(items.len());
      let (targets, rest) = items.split_at(k);
      let shaped = !targets.is_empty()
        && targets.iter().all(|t| {
          if vertex {
            matches!(t, Expr::Integer(_))
          } else {
            pair(t).is_some()
          }
        })
        && rest.iter().all(is_rule);
      shaped.then(|| (targets.to_vec(), rest.to_vec()))
    };
  let specs: Vec<Expr> =
    if split(&spec, true).is_some() || split(&spec, false).is_some() {
      vec![Expr::List(spec.into())]
    } else {
      spec
    };
  let key =
    |e: &Expr| as_rule(e).map(|(k, _)| crate::syntax::expr_to_string(k));
  let merge = |old: &[Expr], new: &[Expr]| -> Vec<Expr> {
    let mut out: Vec<Expr> = old
      .iter()
      .map(|o| {
        new
          .iter()
          .find(|n| key(n).is_some() && key(n) == key(o))
          .unwrap_or(o)
          .clone()
      })
      .collect();
    out.extend(
      new
        .iter()
        .filter(|n| !old.iter().any(|o| key(o).is_some() && key(o) == key(n)))
        .cloned(),
    );
    out
  };
  let directed = option(&gargs[2..], "EdgeDirection")
    .is_some_and(|d| matches!(d, Expr::Identifier(t) if t == "True"))
    || option(opts, "EdgeDirection")
      .is_some_and(|d| matches!(d, Expr::Identifier(t) if t == "True"));
  let sorted = |p: (i128, i128)| {
    if directed {
      p
    } else {
      (p.0.min(p.1), p.0.max(p.1))
    }
  };
  let to_pair = |(a, b): (i128, i128)| {
    Expr::List(vec![Expr::Integer(a), Expr::Integer(b)].into())
  };
  let mut vertex_specs: Vec<(Vec<i128>, Vec<Expr>)> = Vec::new();
  let mut edge_specs: Vec<(Vec<Option<(i128, i128)>>, Vec<Expr>)> = Vec::new();
  for s in &specs {
    let Expr::List(items) = s else { continue };
    if let Some((targets, o)) =
      split(items, true).filter(|(_, o)| !o.is_empty())
    {
      let vs = targets
        .iter()
        .map(|t| match t {
          Expr::Integer(i) => *i,
          _ => 0,
        })
        .collect();
      vertex_specs.push((vs, o));
    } else if let Some((targets, o)) =
      split(items, false).filter(|(_, o)| !o.is_empty())
    {
      let es = targets.iter().map(|t| pair(t).map(sorted)).collect();
      edge_specs.push((es, o));
    }
  }
  let (Expr::List(edges), Expr::List(vertices)) = (&gargs[0], &gargs[1]) else {
    return None;
  };
  let mut new_edges = Vec::with_capacity(edges.len());
  for e in edges {
    let Expr::List(parts) = e else { return None };
    let (head, rest) = parts.split_first()?;
    let head = match pair(head) {
      Some(p) => to_pair(sorted(p)),
      None => head.clone(),
    };
    let found = pair(&head).and_then(|p| {
      edge_specs.iter().enumerate().find_map(|(si, (es, _))| {
        es.iter().position(|q| *q == Some(p)).map(|qi| (si, qi))
      })
    });
    let mut v = vec![head];
    match found {
      Some((si, qi)) => {
        v.extend(merge(rest, &edge_specs[si].1));
        if one {
          edge_specs[si].0[qi] = None;
        }
      }
      None => v.extend(rest.iter().cloned()),
    }
    new_edges.push(Expr::List(v.into()));
  }
  let mut new_vertices = Vec::with_capacity(vertices.len());
  for (i, vx) in vertices.iter().enumerate() {
    let found = vertex_specs
      .iter()
      .find(|(vs, _)| vs.contains(&(i as i128 + 1)));
    match (found, vx) {
      (Some((_, o)), Expr::List(parts)) => {
        let (head, rest) = parts.split_first()?;
        let mut v = vec![head.clone()];
        v.extend(merge(rest, o));
        new_vertices.push(Expr::List(v.into()));
      }
      _ => new_vertices.push(vx.clone()),
    }
  }
  let mut parts = vec![
    Expr::List(new_edges.into()),
    Expr::List(new_vertices.into()),
  ];
  parts.extend(merge(&gargs[2..], opts));
  Some(crate::helpers::call(GRAPH, parts))
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
  Some(circular_graph(*n, edges, false))
}

/// `ExactRandomGraph[n, e]`: `e` distinct edges among `n` vertices, chosen
/// uniformly at random, on the circular embedding.
fn exact_random_graph(args: &[Expr]) -> Option<Expr> {
  use rand::seq::index::sample;
  let [Expr::Integer(n), Expr::Integer(e)] = args else {
    return None;
  };
  if *n < 0 || *n > 2000 || *e < 0 {
    return None;
  }
  let pairs: Vec<(i128, i128)> = (1..=*n)
    .flat_map(|a| (a + 1..=*n).map(move |b| (a, b)))
    .collect();
  if *e as usize > pairs.len() {
    return None;
  }
  let mut picked =
    crate::with_rng(|rng| sample(rng, pairs.len(), *e as usize).into_vec());
  picked.sort_unstable();
  Some(circular_graph(
    *n,
    picked.into_iter().map(|i| pairs[i]).collect(),
    false,
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
  if name == "Combinatorica`FromOrderedPairs" {
    return from_ordered_pairs(args);
  }
  if name == "Combinatorica`SetGraphOptions" {
    return set_graph_options(args);
  }
  if name == "Combinatorica`RandomGraph" {
    return random_graph(args);
  }
  if name == "Combinatorica`ExactRandomGraph" {
    return exact_random_graph(args);
  }
  if name == "Combinatorica`DeleteEdges" {
    return delete_edges(args);
  }
  if name == "Combinatorica`ShowGraph" {
    return show_graph(args, false);
  }
  if name == "Combinatorica`ShowLabeledGraph" {
    return show_graph(args, true);
  }
  let g = CGraph::parse(args.first()?)?;
  match (name, args.len()) {
    ("Combinatorica`IsomorphicQ", 2) => {
      let h = CGraph::parse(&args[1])?;
      Some(Expr::Identifier(
        if isomorphic(&g, &h) { "True" } else { "False" }.to_string(),
      ))
    }
    ("Combinatorica`SpringEmbedding", 1) => spring_embedding(&g, &args[0]),
    // Vertex degrees in decreasing order.
    ("Combinatorica`DegreeSequence", 1) => {
      let mut degrees: Vec<usize> =
        g.adjacency().iter().map(Vec::len).collect();
      degrees.sort_unstable_by(|a, b| b.cmp(a));
      Some(Expr::List(
        degrees
          .into_iter()
          .map(|d| Expr::Integer(d as i128))
          .collect::<Vec<_>>()
          .into(),
      ))
    }
    ("Combinatorica`V", 1) => Some(Expr::Integer(g.n as i128)),
    ("Combinatorica`M", 1) => Some(Expr::Integer(g.edges.len() as i128)),
    ("Combinatorica`Edges" | "Combinatorica`ToUnorderedPairs", 1) => {
      Some(Expr::List(
        g.edges
          .iter()
          .map(|&(a, b)| int_list([a, b]))
          .collect::<Vec<_>>()
          .into(),
      ))
    }
    ("Combinatorica`ToAdjacencyMatrix", 1) => {
      let mut m = vec![vec![0i128; g.n]; g.n];
      for &(a, b) in &g.edges {
        m[a][b] += 1;
        if a != b && !g.directed {
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
    // The vertices within `d` edges of `v` (including `v`), in increasing
    // order.
    ("Combinatorica`Neighborhood", 3) => {
      let start = vertex_arg(&g, &args[1])?;
      let Expr::Integer(d) = &args[2] else {
        return None;
      };
      let (_, level, _) = g.bfs(start);
      Some(int_list(
        (0..g.n).filter(|&v| level[v].is_some_and(|l| (l as i128) <= *d)),
      ))
    }
    // Edge-count distances between all vertex pairs, Infinity where no
    // path joins them (GraphUtilities` reading a Combinatorica graph).
    ("Combinatorica`GraphDistanceMatrix", 1) => Some(Expr::List(
      (0..g.n)
        .map(|start| {
          let (_, level, _) = g.bfs(start);
          Expr::List(
            level
              .into_iter()
              .map(|d| match d {
                Some(d) => Expr::Integer(d as i128),
                None => Expr::Identifier("Infinity".to_string()),
              })
              .collect::<Vec<_>>()
              .into(),
          )
        })
        .collect::<Vec<_>>()
        .into(),
    )),
    // Shortest-path lengths between all vertex pairs: Floyd–Warshall over
    // the `EdgeWeight`s (1 by default) in machine reals, as Combinatorica's
    // compiled kernel runs it, with integral lengths turned back into
    // integers (its `Zap`) and Infinity where no path joins two vertices.
    ("Combinatorica`AllPairsShortestPath", 1) => {
      let Expr::FunctionCall { args: gargs, .. } = &args[0] else {
        return None;
      };
      let Expr::List(raw_edges) = &gargs[0] else {
        return None;
      };
      let mut d = vec![vec![f64::INFINITY; g.n]; g.n];
      for (i, row) in d.iter_mut().enumerate() {
        row[i] = 0.0;
      }
      for (&(u, v), raw) in g.edges.iter().zip(raw_edges.iter()) {
        let Expr::List(parts) = raw else { return None };
        let w = match option(&parts[1..], "EdgeWeight") {
          Some(w) => crate::functions::math_ast::try_eval_to_f64(w)?,
          None => 1.0,
        };
        if u != v {
          d[u][v] = d[u][v].min(w);
          if !g.directed {
            d[v][u] = d[v][u].min(w);
          }
        }
      }
      for k in 0..g.n {
        for i in 0..g.n {
          for j in 0..g.n {
            d[i][j] = d[i][j].min(d[i][k] + d[k][j]);
          }
        }
      }
      let zap = |x: f64| {
        if x.is_infinite() {
          Expr::Identifier("Infinity".to_string())
        } else if (x.round() - x).abs() < 1e-10 {
          Expr::Integer(x.round() as i128)
        } else {
          Expr::Real(x)
        }
      };
      Some(Expr::List(
        d.into_iter()
          .map(|row| {
            Expr::List(row.into_iter().map(zap).collect::<Vec<_>>().into())
          })
          .collect::<Vec<_>>()
          .into(),
      ))
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
  // The graph's own options, overridden by the call's Combinatorica ones;
  // the call's other options (ImageSize, …) go on to Graphics.
  const OWN: [&str; 9] = [
    "EdgeStyle",
    "VertexStyle",
    "VertexNumber",
    "VertexNumberPosition",
    "VertexNumberColor",
    "VertexColor",
    "EdgeColor",
    "EdgeDirection",
    "VertexLabel",
  ];
  let own = |o: &Expr| {
    crate::evaluator::dispatch::list_operations::as_rule(o)
      .is_some_and(|(k, _)| OWN.iter().any(|n| symbol_is(k, n)))
  };
  let mut gopts: Vec<Expr> = extra.iter().filter(|o| own(o)).cloned().collect();
  gopts.extend(gargs[2..].iter().cloned());
  let gopts = &gopts[..];
  let extra: Vec<Expr> = extra.iter().filter(|o| !own(o)).cloned().collect();
  let labeled = labeled
    || option(gopts, "VertexNumber")
      .is_some_and(|v| matches!(v, Expr::Identifier(t) if t == "True"));
  let directed = option(gopts, "EdgeDirection")
    .is_some_and(|d| matches!(d, Expr::Identifier(t) if t == "True"));
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
  // An edge style that is a graphics directive (Thin, Arrowheads[…], a
  // colour) applies to every edge.
  if let Some(style) = option(gopts, "EdgeStyle")
    && !matches!(style, Expr::Identifier(n) if n == "Normal")
  {
    prims.push(style.clone());
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
      if directed { "Arrow" } else { "Line" },
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
  let centered = option(gopts, "VertexNumberPosition")
    .is_some_and(|p| matches!(p, Expr::Identifier(c) if c == "Center"));
  let (dx, dy) = option(gopts, "VertexNumberPosition")
    .and_then(real_pair)
    .unwrap_or((0.0, 0.0));
  let label_color = option(gopts, "VertexNumberColor").cloned();
  for (i, (v, &(x, y))) in vertices.iter().zip(&coords).enumerate() {
    let Expr::List(parts) = v else { return None };
    let disk = crate::helpers::call("Disk", vec![pt(x, y), Expr::Real(radius)]);
    match option(&parts[1..], "VertexColor") {
      Some(color) => prims.push(Expr::List(vec![color.clone(), disk].into())),
      None => prims.push(disk),
    }
    if labeled {
      let lift = if centered { 0.0 } else { 2.0 * radius };
      let text = crate::helpers::call(
        "Text",
        vec![
          Expr::Integer(i as i128 + 1),
          pt(x + dx * span, y + dy * span + lift),
        ],
      );
      match &label_color {
        Some(c) => prims.push(Expr::List(vec![c.clone(), text].into())),
        None => prims.push(text),
      }
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
