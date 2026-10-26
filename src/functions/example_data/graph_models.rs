//! Random graph models for generated networks: a configuration model over a
//! heavy-tailed degree sequence with prescribed minimum, maximum and mean
//! degree, simplified and topped up to an exact edge count.

use std::collections::HashSet;

use super::rng::Rng;

/// The target shape of a generated graph.
pub struct GraphSpec {
  pub vertices: usize,
  pub edges: usize,
  pub directed: bool,
  /// Whether multi-edges and self-loops are allowed.
  pub multigraph: bool,
  pub loops: bool,
  pub min_degree: f64,
  pub max_degree: f64,
}

/// A degree sequence of `n` values in `[min, max]` summing to `total`,
/// heavy-tailed (most vertices near the minimum, a few hubs), with the
/// maximum attained.
pub fn degree_sequence(
  rng: &mut Rng,
  n: usize,
  min: usize,
  max: usize,
  total: usize,
) -> Vec<usize> {
  if n == 0 {
    return Vec::new();
  }
  let max = max.max(min);
  let mean = total as f64 / n as f64;
  // u^alpha has mean 1/(alpha+1); pick alpha so the expected mean matches.
  let span = (max - min) as f64;
  let alpha = if mean > min as f64 && span > 0.0 {
    (span / (mean - min as f64) - 1.0).max(0.05)
  } else {
    1.0
  };
  let mut d: Vec<usize> = (0..n)
    .map(|_| min + (span * rng.uniform().powf(alpha)).round() as usize)
    .collect();
  d[0] = max;
  // Correct the sum one step at a time on random vertices within bounds.
  let mut sum: usize = d.iter().sum();
  let mut guard = 0usize;
  while sum != total && guard < 50 * n + 1000 {
    guard += 1;
    let i = 1 + rng.below(n.saturating_sub(1).max(1)).min(n - 1);
    let i = if n == 1 { 0 } else { i };
    if sum < total && d[i] < max {
      d[i] += 1;
      sum += 1;
    } else if sum > total && d[i] > min {
      d[i] -= 1;
      sum -= 1;
    }
  }
  rng.shuffle(&mut d);
  d
}

/// Edges (vertex index pairs) with exactly `spec.edges` entries.
pub fn generate(rng: &mut Rng, spec: &GraphSpec) -> Vec<(usize, usize)> {
  let n = spec.vertices;
  let m = spec.edges;
  if n == 0 || m == 0 {
    return Vec::new();
  }
  let min = spec.min_degree.round().max(0.0) as usize;
  let max = (spec.max_degree.round() as usize)
    .max(min)
    .min(if spec.multigraph { usize::MAX } else { n });
  let mut edges: Vec<(usize, usize)> = Vec::with_capacity(m);
  let mut seen: HashSet<(usize, usize)> = HashSet::with_capacity(m * 2);
  let key = |a: usize, b: usize| {
    if spec.directed || a <= b {
      (a, b)
    } else {
      (b, a)
    }
  };
  let mut try_add =
    |a: usize, b: usize, edges: &mut Vec<(usize, usize)>| -> bool {
      if a == b && !spec.loops {
        return false;
      }
      let k = key(a, b);
      if !spec.multigraph && !seen.insert(k) {
        return false;
      }
      edges.push((a, b));
      true
    };
  if spec.directed {
    let out_d = degree_sequence(rng, n, min / 2, max, m);
    let in_d = degree_sequence(rng, n, min / 2, max, m);
    let mut outs: Vec<usize> = out_d
      .iter()
      .enumerate()
      .flat_map(|(v, &k)| std::iter::repeat_n(v, k))
      .collect();
    let mut ins: Vec<usize> = in_d
      .iter()
      .enumerate()
      .flat_map(|(v, &k)| std::iter::repeat_n(v, k))
      .collect();
    rng.shuffle(&mut outs);
    rng.shuffle(&mut ins);
    for (a, b) in outs.into_iter().zip(ins) {
      try_add(a, b, &mut edges);
    }
  } else {
    let d = degree_sequence(rng, n, min, max, 2 * m);
    let mut stubs: Vec<usize> = d
      .iter()
      .enumerate()
      .flat_map(|(v, &k)| std::iter::repeat_n(v, k))
      .collect();
    rng.shuffle(&mut stubs);
    for pair in stubs.chunks(2) {
      if let [a, b] = pair {
        try_add(*a, *b, &mut edges);
      }
    }
  }
  // Top up edges lost to simplification, attaching preferentially to
  // vertices that already have edges (keeps the degree profile).
  let mut guard = 0usize;
  while edges.len() < m && guard < 100 * m + 1000 {
    guard += 1;
    let a = if edges.is_empty() || rng.chance(0.5) {
      rng.below(n)
    } else {
      let e = edges[rng.below(edges.len())];
      if rng.chance(0.5) { e.0 } else { e.1 }
    };
    let b = rng.below(n);
    try_add(a, b, &mut edges);
  }
  edges.truncate(m);
  edges
}
