//! `FindShortestTour[points]`, `FindShortestTour[points, s, t]`.
//!
//! Up to `EXACT_MAX` points the tour is optimal (Held–Karp dynamic
//! programming); larger inputs get a nearest-neighbour tour improved by
//! neighbour-list 2-opt and Or-opt until no move shortens it. The optimisation
//! runs on machine distances; the reported length is the exact sum of the
//! chosen legs (`2 + 2*Sqrt[2]` for exact points).

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::dendrogram::distance_matrix;
use crate::functions::list_helpers_ast::apply_func_to_two_args;
use crate::helpers::unevaluated;
use crate::syntax::Expr;

/// Largest instance solved exactly.
const EXACT_MAX: usize = 12;

/// Neighbour-list size for the local search.
const NEIGHBOURS: usize = 12;

pub fn find_shortest_tour_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let is_option =
    |a: &Expr| matches!(a, Expr::Rule { .. } | Expr::RuleDelayed { .. });
  let n_positional = args.iter().take_while(|a| !is_option(a)).count();
  let call_str =
    || crate::syntax::expr_to_string(&unevaluated("FindShortestTour", args));
  match n_positional {
    0 => {
      crate::emit_message(&format!(
        "FindShortestTour::argb: FindShortestTour called with {} arguments; between 1 and 3 arguments are expected.",
        args.len()
      ));
      return Ok(unevaluated("FindShortestTour", args));
    }
    2 => {
      crate::emit_message(
        "FindShortestTour::argt: FindShortestTour called with 2 arguments; 1 or 3 arguments are expected.",
      );
      return Ok(unevaluated("FindShortestTour", args));
    }
    _ => {}
  }
  let max_positional = if n_positional == 1 { 1 } else { 3 };
  if let Some(extra) = args[max_positional..].iter().find(|a| !is_option(a)) {
    crate::emit_message(&format!(
      "FindShortestTour::nonopt: Options expected (instead of {}) beyond position {max_positional} in {}. An option must be a rule or a list of rules.",
      crate::syntax::expr_to_string(extra),
      call_str()
    ));
    return Ok(unevaluated("FindShortestTour", args));
  }
  let positional = &args[..max_positional];
  let options = &args[max_positional..];
  if let Some(result) = graph_tour(&positional[0], positional.len() == 1) {
    return Ok(result);
  }
  let Expr::List(points) = &positional[0] else {
    return Ok(unevaluated("FindShortestTour", args));
  };
  let n = points.len();

  let mut distance_fn = None;
  for opt in options {
    if let Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } = opt
      && matches!(pattern.as_ref(), Expr::Identifier(s) if s == "DistanceFunction")
      && !matches!(replacement.as_ref(), Expr::Identifier(s) if s == "Automatic")
    {
      distance_fn = Some(replacement.as_ref().clone());
    }
  }

  // Endpoints of an open path (1-based in the input); `s == t` asks for the
  // closed tour.
  let endpoints = if positional.len() == 3 {
    let index = |e: &Expr| match e {
      Expr::Integer(k) if *k >= 1 && (*k as usize) <= n => {
        Some(*k as usize - 1)
      }
      _ => None,
    };
    match (index(&positional[1]), index(&positional[2])) {
      (Some(s), Some(t)) if s == t => None,
      (Some(s), Some(t)) => Some((s, t)),
      _ => return Ok(Expr::List(Vec::new().into())),
    }
  } else {
    None
  };

  if n == 0 {
    return Ok(Expr::List(Vec::new().into()));
  }
  let Some(dist) = distance_matrix(points, distance_fn.as_ref()) else {
    return Ok(unevaluated("FindShortestTour", args));
  };

  let order = match endpoints {
    None => closed_tour(&dist),
    Some((s, t)) => open_path(&dist, s, t),
  };

  // Exact length: the sum of the legs as the distance function states them.
  let head = distance_fn.unwrap_or_else(|| {
    if points.iter().all(|p| matches!(p, Expr::String(_))) {
      Expr::Identifier("EditDistance".into())
    } else {
      Expr::Identifier("EuclideanDistance".into())
    }
  });
  let mut legs = Vec::with_capacity(order.len());
  for w in order.windows(2) {
    legs.push(apply_func_to_two_args(&head, &points[w[0]], &points[w[1]])?);
  }
  let length = if legs.is_empty() {
    Expr::Integer(0)
  } else {
    evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Plus".into(),
      args: legs.into(),
    })?
  };
  let indices: Vec<Expr> = order
    .iter()
    .map(|&i| Expr::Integer(i as i128 + 1))
    .collect();
  Ok(Expr::List(vec![length, Expr::List(indices.into())].into()))
}

/// Search budget (visited partial tours) for the graph tour.
const GRAPH_TOUR_NODE_LIMIT: u64 = 20_000_000;

/// `FindShortestTour[graph]`: the cheapest closed tour that follows the
/// graph's edges and visits every vertex once, as `{cost, {v1, …, v1}}`
/// (edge weights from `EdgeWeight`, default 1), or `{}` when the graph has
/// no such tour. Returns `None` when `graph` is not a graph.
fn graph_tour(graph: &Expr, single: bool) -> Option<Expr> {
  let Expr::FunctionCall { name, args: gargs } = graph else {
    return None;
  };
  if !single || (name != "Graph" && name != "Graph3D") {
    return None;
  }
  // Graph3D[Graph[…]] wraps a plain graph; Graph3D[vertices, edges, …] has
  // the same layout as Graph.
  if name == "Graph3D"
    && let [inner @ Expr::FunctionCall { name: n, .. }] = gargs.as_slice()
    && n == "Graph"
  {
    return graph_tour(inner, true);
  }
  let (vertices, edges) = crate::functions::graph::fc_parse_input(graph)?;
  let n = vertices.len();
  if n == 0 {
    return Some(Expr::List(Vec::new().into()));
  }
  let weights: Vec<Expr> = gargs
    .iter()
    .find_map(|a| match a {
      Expr::Rule {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "EdgeWeight") => {
        match replacement.as_ref() {
          Expr::List(ws) => Some(ws.iter().cloned().collect()),
          _ => None,
        }
      }
      _ => None,
    })
    .unwrap_or_default();

  let key = |v: &Expr| crate::syntax::expr_to_string(v);
  let rank: std::collections::HashMap<String, usize> = vertices
    .iter()
    .enumerate()
    .rev()
    .map(|(i, v)| (key(v), i))
    .collect();
  // cost[i][j]: cheapest edge i → j; weight_expr keeps its exact value.
  let mut cost = vec![vec![f64::INFINITY; n]; n];
  let mut exact: Vec<Vec<Option<Expr>>> = vec![vec![None; n]; n];
  for (k, e) in edges.iter().enumerate() {
    let Some((u, v, directed)) = crate::functions::graph::edge_endpoints(e)
    else {
      continue;
    };
    let (Some(&i), Some(&j)) = (rank.get(&key(&u)), rank.get(&key(&v))) else {
      continue;
    };
    if i == j {
      continue;
    }
    let w_expr = weights.get(k).cloned().unwrap_or(Expr::Integer(1));
    // `Automatic` is the default weight 1.
    let w_expr = if matches!(&w_expr, Expr::Identifier(s) if s == "Automatic") {
      Expr::Integer(1)
    } else {
      w_expr
    };
    let w = crate::functions::math_ast::try_eval_to_f64(&w_expr)?;
    let mut put = |a: usize, b: usize| {
      if w < cost[a][b] {
        cost[a][b] = w;
        exact[a][b] = Some(w_expr.clone());
      }
    };
    put(i, j);
    if !directed {
      put(j, i);
    }
  }

  let order = min_cost_cycle(&cost);
  let Some(mut order) = order else {
    return Some(Expr::List(Vec::new().into()));
  };
  // Of the two directions the one whose second stop has the higher index is
  // reported, as for point sets.
  if n > 2 && order[1] < order[n - 1] {
    let rev_ok = (1..n).all(|k| {
      let (a, b) = (order[(n - k) % n], order[n - k - 1]);
      cost[a][b].is_finite()
    }) && cost[order[0]][order[n - 1]].is_finite();
    if rev_ok {
      order[1..].reverse();
    }
  }
  let mut legs = Vec::with_capacity(n);
  for k in 0..n {
    let (a, b) = (order[k], order[(k + 1) % n]);
    legs.push(exact[a][b].clone()?);
  }
  let total = evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Plus".into(),
    args: legs.into(),
  })
  .ok()?;
  let mut tour: Vec<Expr> =
    order.iter().map(|&i| vertices[i].clone()).collect();
  tour.push(vertices[order[0]].clone());
  Some(Expr::List(vec![total, Expr::List(tour.into())].into()))
}

/// The cheapest Hamiltonian cycle through `cost` (infinite = no edge)
/// starting at vertex 0, by depth-first branch and bound; `None` if there is
/// none. The visit order of ties is deterministic (lowest index first).
fn min_cost_cycle(cost: &[Vec<f64>]) -> Option<Vec<usize>> {
  let n = cost.len();
  if n < 3 {
    return None;
  }
  // Cheapest edge leaving each vertex bounds the remaining cost from below.
  let min_out: Vec<f64> = cost
    .iter()
    .map(|row| row.iter().copied().fold(f64::INFINITY, f64::min))
    .collect();
  if min_out.iter().any(|m| !m.is_finite()) {
    return None;
  }
  struct Search<'a> {
    cost: &'a [Vec<f64>],
    min_out: &'a [f64],
    path: Vec<usize>,
    used: Vec<bool>,
    best: f64,
    best_path: Option<Vec<usize>>,
    nodes: u64,
  }
  fn go(s: &mut Search, so_far: f64, remaining_bound: f64) {
    s.nodes += 1;
    if s.nodes > GRAPH_TOUR_NODE_LIMIT {
      return;
    }
    let n = s.cost.len();
    let last = *s.path.last().unwrap();
    if s.path.len() == n {
      let total = so_far + s.cost[last][0];
      if total < s.best - 1e-12 {
        s.best = total;
        s.best_path = Some(s.path.clone());
      }
      return;
    }
    let mut next: Vec<usize> = (0..n)
      .filter(|&j| !s.used[j] && s.cost[last][j].is_finite())
      .collect();
    next.sort_by(|&a, &b| s.cost[last][a].total_cmp(&s.cost[last][b]));
    for j in next {
      let c = so_far + s.cost[last][j];
      let bound = remaining_bound - s.min_out[last];
      if c + bound >= s.best - 1e-12 {
        continue;
      }
      s.used[j] = true;
      s.path.push(j);
      go(s, c, bound);
      s.path.pop();
      s.used[j] = false;
    }
  }
  let mut s = Search {
    cost,
    min_out: &min_out,
    path: vec![0],
    used: {
      let mut u = vec![false; n];
      u[0] = true;
      u
    },
    best: f64::INFINITY,
    best_path: None,
    nodes: 0,
  };
  let total_bound: f64 = min_out.iter().sum();
  go(&mut s, 0.0, total_bound);
  s.best_path
}

/// A closed tour through every point, starting and ending at point 0. Of the
/// two directions the one whose second stop has the higher index is
/// reported, as Wolfram Language does.
fn closed_tour(dist: &[Vec<f64>]) -> Vec<usize> {
  let n = dist.len();
  let mut tour: Vec<usize> = if n <= 2 {
    (0..n).collect()
  } else if n <= EXACT_MAX {
    held_karp(dist, 0, None)
  } else {
    let mut tour = nearest_neighbour_tour(dist);
    optimise_tour(dist, &mut tour);
    let start = tour.iter().position(|&c| c == 0).unwrap_or(0);
    tour.rotate_left(start);
    tour
  };
  if tour.len() > 2 && tour[1] < tour[tour.len() - 1] {
    tour[1..].reverse();
  }
  if n > 1 {
    tour.push(0);
  }
  tour
}

/// The shortest path from `s` to `t` through every point.
fn open_path(dist: &[Vec<f64>], s: usize, t: usize) -> Vec<usize> {
  let n = dist.len();
  if n <= EXACT_MAX {
    return held_karp(dist, s, Some(t));
  }
  // Pin the edge t–s with a large negative weight so the local search keeps
  // it, then cut the tour there.
  let big = -1.0e6 * dist.iter().flatten().fold(1.0_f64, |m, &d| m.max(d));
  let mut pinned = dist.to_vec();
  pinned[s][t] = big;
  pinned[t][s] = big;
  let mut tour = nearest_neighbour_tour_from(&pinned, s);
  optimise_tour(&pinned, &mut tour);
  let start = tour.iter().position(|&c| c == s).unwrap_or(0);
  tour.rotate_left(start);
  if tour.last() != Some(&t) {
    tour[1..].reverse();
  }
  tour
}

/// Optimal order by dynamic programming over subsets. With `end` set the
/// result is a path from `start` to `end`, otherwise a closed tour (returned
/// without the repeated start).
fn held_karp(
  dist: &[Vec<f64>],
  start: usize,
  end: Option<usize>,
) -> Vec<usize> {
  let n = dist.len();
  let full = 1usize << n;
  let mut cost = vec![f64::INFINITY; full * n];
  let mut parent = vec![usize::MAX; full * n];
  cost[(1 << start) * n + start] = 0.0;
  for mask in 0..full {
    if mask & (1 << start) == 0 {
      continue;
    }
    for last in 0..n {
      let c = cost[mask * n + last];
      if !c.is_finite() || mask & (1 << last) == 0 {
        continue;
      }
      for next in 0..n {
        if mask & (1 << next) != 0 {
          continue;
        }
        // The fixed end is entered last.
        if end == Some(next) && (mask | (1 << next)) != full - 1 {
          continue;
        }
        let nm = mask | (1 << next);
        let nc = c + dist[last][next];
        if nc < cost[nm * n + next] - 1e-12 {
          cost[nm * n + next] = nc;
          parent[nm * n + next] = last;
        }
      }
    }
  }
  let all = full - 1;
  let mut last = match end {
    Some(e) => e,
    None => (0..n)
      .filter(|&l| l != start || n == 1)
      .min_by(|&a, &b| {
        let ca = cost[all * n + a] + dist[a][start];
        let cb = cost[all * n + b] + dist[b][start];
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
      })
      .unwrap_or(start),
  };
  let mut order = Vec::with_capacity(n);
  let mut mask = all;
  while last != usize::MAX {
    order.push(last);
    let p = parent[mask * n + last];
    mask &= !(1 << last);
    last = p;
  }
  order.reverse();
  order
}

fn nearest_neighbour_tour(dist: &[Vec<f64>]) -> Vec<usize> {
  nearest_neighbour_tour_from(dist, 0)
}

fn nearest_neighbour_tour_from(dist: &[Vec<f64>], start: usize) -> Vec<usize> {
  let n = dist.len();
  let mut visited = vec![false; n];
  let mut tour = Vec::with_capacity(n);
  let mut cur = start;
  visited[cur] = true;
  tour.push(cur);
  for _ in 1..n {
    let mut best = usize::MAX;
    for j in 0..n {
      if !visited[j] && (best == usize::MAX || dist[cur][j] < dist[cur][best]) {
        best = j;
      }
    }
    visited[best] = true;
    tour.push(best);
    cur = best;
  }
  tour
}

fn tour_length(dist: &[Vec<f64>], tour: &[usize]) -> f64 {
  let n = tour.len();
  (0..n).map(|i| dist[tour[i]][tour[(i + 1) % n]]).sum()
}

/// Iterated local search: a 2-opt/Or-opt local optimum is perturbed by a
/// double-bridge move and re-optimised, keeping the result when it is
/// shorter. The kicks come from a fixed-seed generator so the tour is
/// reproducible.
fn optimise_tour(dist: &[Vec<f64>], tour: &mut Vec<usize>) {
  let neighbours = neighbour_lists(dist);
  improve_tour(dist, &neighbours, tour);
  let n = tour.len();
  if n < 8 {
    return;
  }
  let kicks = (3_000_000 / n).clamp(100, 3000);
  let mut best_len = tour_length(dist, tour);
  let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
  let mut next = |bound: usize| {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    (state % bound as u64) as usize
  };
  for _ in 0..kicks {
    // Cut points 0 < p1 < p2 < p3 < n: A B C D → A C B D.
    let mut cuts = [1 + next(n - 1), 1 + next(n - 1), 1 + next(n - 1)];
    cuts.sort_unstable();
    let [p1, p2, p3] = cuts;
    if p1 == p2 || p2 == p3 {
      continue;
    }
    let mut cand = Vec::with_capacity(n);
    cand.extend_from_slice(&tour[..p1]);
    cand.extend_from_slice(&tour[p2..p3]);
    cand.extend_from_slice(&tour[p1..p2]);
    cand.extend_from_slice(&tour[p3..]);
    improve_tour(dist, &neighbours, &mut cand);
    let len = tour_length(dist, &cand);
    if len < best_len - 1e-10 {
      best_len = len;
      *tour = cand;
    }
  }
}

/// Each city's `NEIGHBOURS` nearest other cities, closest first. Uses
/// `select_nth_unstable_by` so building the lists is O(n²), not O(n² log n).
fn neighbour_lists(dist: &[Vec<f64>]) -> Vec<Vec<usize>> {
  let n = dist.len();
  let by_dist = |i: usize| {
    move |a: &usize, b: &usize| {
      dist[i][*a]
        .partial_cmp(&dist[i][*b])
        .unwrap_or(std::cmp::Ordering::Equal)
    }
  };
  (0..n)
    .map(|i| {
      let mut others: Vec<usize> = (0..n).filter(|&j| j != i).collect();
      if others.len() > NEIGHBOURS {
        others.select_nth_unstable_by(NEIGHBOURS, by_dist(i));
        others.truncate(NEIGHBOURS);
      }
      others.sort_by(by_dist(i));
      others
    })
    .collect()
}

/// 2-opt and Or-opt moves over each city's nearest neighbours, repeated until
/// no move shortens the tour.
fn improve_tour(
  dist: &[Vec<f64>],
  neighbours: &[Vec<usize>],
  tour: &mut [usize],
) {
  let n = tour.len();
  if n < 4 {
    return;
  }
  let mut pos = vec![0usize; n];
  let sync = |tour: &[usize], pos: &mut [usize]| {
    for (i, &c) in tour.iter().enumerate() {
      pos[c] = i;
    }
  };
  sync(tour, &mut pos);
  const EPS: f64 = 1e-10;
  let mut improved = true;
  let mut rounds = 0;
  while improved && rounds < 1000 {
    improved = false;
    rounds += 1;
    // 2-opt: replace edges (a, a+1) and (c, c+1) by (a, c) and (a+1, c+1).
    for i in 0..n {
      let a = tour[i];
      let a_next = tour[(i + 1) % n];
      let d_a = dist[a][a_next];
      for &c in &neighbours[a] {
        if dist[a][c] >= d_a {
          break;
        }
        let j = pos[c];
        let c_next = tour[(j + 1) % n];
        if c == a_next || c_next == a {
          continue;
        }
        let delta = dist[a][c] + dist[a_next][c_next] - d_a - dist[c][c_next];
        if delta < -EPS {
          let (lo, hi) = if i < j { (i + 1, j) } else { (j + 1, i) };
          tour[lo..=hi].reverse();
          sync(tour, &mut pos);
          improved = true;
          break;
        }
      }
    }
    // Or-opt: move a run of 1–3 cities between c and its successor.
    for seg_len in 1..=3usize {
      if seg_len + 2 >= n {
        break;
      }
      for i in 0..n {
        let first = tour[i];
        let last = tour[(i + seg_len - 1) % n];
        let prev = tour[(i + n - 1) % n];
        let next = tour[(i + seg_len) % n];
        let removed = dist[prev][first] + dist[last][next] - dist[prev][next];
        let in_seg = |c: usize| {
          let k = (pos[c] + n - i) % n;
          k < seg_len
        };
        for &c in &neighbours[first] {
          if in_seg(c) {
            continue;
          }
          let c_next = tour[(pos[c] + 1) % n];
          if in_seg(c_next) {
            continue;
          }
          // Insert reversed (c, last … first, c_next) or forward
          // (c, first … last, c_next).
          let fwd = dist[c][first] + dist[last][c_next] - dist[c][c_next];
          let rev = dist[c][last] + dist[first][c_next] - dist[c][c_next];
          let (gain, reversed) = if fwd <= rev {
            (removed - fwd, false)
          } else {
            (removed - rev, true)
          };
          if gain > EPS {
            let mut seg: Vec<usize> =
              (0..seg_len).map(|k| tour[(i + k) % n]).collect();
            if reversed {
              seg.reverse();
            }
            let rest: Vec<usize> = (0..n - seg_len)
              .map(|k| tour[(i + seg_len + k) % n])
              .collect();
            let at = rest.iter().position(|&x| x == c).unwrap() + 1;
            let mut new_tour = Vec::with_capacity(n);
            new_tour.extend_from_slice(&rest[..at]);
            new_tour.extend_from_slice(&seg);
            new_tour.extend_from_slice(&rest[at..]);
            tour.copy_from_slice(&new_tour);
            sync(tour, &mut pos);
            improved = true;
            break;
          }
        }
      }
    }
  }
}
