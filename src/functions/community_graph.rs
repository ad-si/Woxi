//! `FindGraphCommunities` and `CommunityGraphPlot`: the communities of a
//! graph (by modularity, Wolfram's default method) and the plot that lays
//! each community out on its own, colours its vertices and rings it with a
//! smooth boundary.

use crate::InterpreterError;
use crate::helpers::{call, call1, id_expr, rule_expr, str_expr};
use crate::syntax::Expr;

/// The vertices and the (undirected) edges of a `Graph[…]`, as vertex
/// indices. `None` for anything that is not a graph.
fn graph_parts(g: &Expr) -> Option<(Vec<Expr>, Vec<(usize, usize)>)> {
  let Expr::FunctionCall { name, args } = g else {
    return None;
  };
  if name != "Graph" {
    return None;
  }
  let Expr::List(vertices) = args.first()? else {
    return None;
  };
  let (_, pairs) = crate::functions::graph::parse_graph_pairs(g)?;
  Some((
    vertices.to_vec(),
    pairs.into_iter().map(|(a, b, _)| (a, b)).collect(),
  ))
}

/// Communities of maximal modularity, found the Louvain way: each vertex
/// moves to the neighbouring community it raises the modularity most, in
/// vertex order, until none moves; the communities then become single
/// vertices of a smaller graph, and the moving starts over. Each community
/// lists its vertex indices in order; the communities come largest first,
/// ties by their first vertex — the order `FindGraphCommunities` returns.
pub(crate) fn modularity_communities(
  n: usize,
  edges: &[(usize, usize)],
) -> Vec<Vec<usize>> {
  // The weighted graph being clustered, and which original vertices each
  // of its nodes stands for.
  let mut members: Vec<Vec<usize>> = (0..n).map(|v| vec![v]).collect();
  let mut weights: Vec<std::collections::BTreeMap<usize, f64>> =
    vec![std::collections::BTreeMap::default(); n];
  for &(a, b) in edges {
    if a == b {
      continue;
    }
    *weights[a].entry(b).or_insert(0.0) += 1.0;
    *weights[b].entry(a).or_insert(0.0) += 1.0;
  }
  let total: f64 = weights.iter().flat_map(|w| w.values()).sum::<f64>();
  if total == 0.0 {
    return (0..n).map(|v| vec![v]).collect();
  }
  // `total` counts every edge twice, i.e. it is 2m.
  let two_m = total;
  // Weight of the edges inside each node (from earlier aggregation).
  let mut self_weight = vec![0.0; n];
  loop {
    let nodes = members.len();
    let degree: Vec<f64> = (0..nodes)
      .map(|i| weights[i].values().sum::<f64>() + self_weight[i])
      .collect();
    let mut community: Vec<usize> = (0..nodes).collect();
    let mut tot: Vec<f64> = degree.clone();
    let mut moved_any = false;
    loop {
      let mut moved = false;
      for i in 0..nodes {
        let own = community[i];
        tot[own] -= degree[i];
        // Links from `i` into each neighbouring community.
        let mut links: std::collections::BTreeMap<usize, f64> =
          std::collections::BTreeMap::default();
        for (&j, &w) in &weights[i] {
          *links.entry(community[j]).or_insert(0.0) += w;
        }
        let gain = |c: usize| {
          links.get(&c).copied().unwrap_or(0.0) - tot[c] * degree[i] / two_m
        };
        let mut best = own;
        let mut best_gain = gain(own);
        for &c in links.keys() {
          let g = gain(c);
          if g > best_gain + 1e-12 {
            best = c;
            best_gain = g;
          }
        }
        tot[best] += degree[i];
        if best != own {
          community[i] = best;
          moved = true;
          moved_any = true;
        }
      }
      if !moved {
        break;
      }
    }
    if !moved_any {
      break;
    }
    // Aggregate: one node per community.
    let mut index: std::collections::BTreeMap<usize, usize> =
      std::collections::BTreeMap::default();
    for &c in &community {
      let next = index.len();
      index.entry(c).or_insert(next);
    }
    let k = index.len();
    let mut new_members: Vec<Vec<usize>> = vec![Vec::new(); k];
    let mut new_weights: Vec<std::collections::BTreeMap<usize, f64>> =
      vec![std::collections::BTreeMap::default(); k];
    let mut new_self = vec![0.0; k];
    for i in 0..nodes {
      let ci = index[&community[i]];
      new_members[ci].extend(members[i].iter().copied());
      new_self[ci] += self_weight[i];
      for (&j, &w) in &weights[i] {
        let cj = index[&community[j]];
        if ci == cj {
          new_self[ci] += w;
        } else {
          *new_weights[ci].entry(cj).or_insert(0.0) += w;
        }
      }
    }
    members = new_members;
    weights = new_weights;
    self_weight = new_self;
  }
  for m in &mut members {
    m.sort_unstable();
  }
  members.sort_by(|a, b| b.len().cmp(&a.len()).then(a[0].cmp(&b[0])));
  members
}

/// `FindGraphCommunities[g]`.
pub fn find_graph_communities_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "FindGraphCommunities".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() != 1 {
    return unevaluated();
  }
  let Some((vertices, edges)) = graph_parts(&args[0]) else {
    return unevaluated();
  };
  let communities = modularity_communities(vertices.len(), &edges);
  Ok(Expr::List(
    communities
      .iter()
      .map(|c| {
        Expr::List(
          c.iter()
            .map(|&v| vertices[v].clone())
            .collect::<Vec<_>>()
            .into(),
        )
      })
      .collect::<Vec<_>>()
      .into(),
  ))
}

/// The colour `CommunityGraphPlot` gives its `k`-th community: Wolfram's
/// seven fixed hues, then hues a golden-ratio turn apart.
fn community_color(k: usize) -> Expr {
  const FIXED: [(f64, f64, f64); 7] = [
    (0.0, 1.0, 0.8),
    (0.14, 1.0, 0.9),
    (0.8, 0.6, 0.8),
    (0.07, 1.0, 1.0),
    (0.2, 1.0, 0.6),
    (0.1, 0.6, 0.7),
    (0.5, 1.0, 0.7),
  ];
  let (h, s, b) = FIXED.get(k).copied().unwrap_or_else(|| {
    let golden = f64::midpoint(1.0, 5f64.sqrt());
    (0.2 + (k - FIXED.len()) as f64 * golden, 1.0, 0.8)
  });
  call("Hue", vec![Expr::Real(h), Expr::Real(s), Expr::Real(b)])
}

fn point(p: (f64, f64)) -> Expr {
  Expr::List(vec![Expr::Real(p.0), Expr::Real(p.1)].into())
}

/// The layout of the vertices `verts` (indices) joined by `edges`, as the
/// plain `Graph` of them would draw it.
fn sub_layout(
  vertices: &[Expr],
  verts: &[usize],
  edges: &[(usize, usize)],
) -> Vec<(f64, f64)> {
  if verts.len() == 1 {
    return vec![(0.0, 0.0)];
  }
  let vlist: Vec<Expr> = verts.iter().map(|&v| vertices[v].clone()).collect();
  let elist: Vec<Expr> = edges
    .iter()
    .filter(|(a, b)| verts.contains(a) && verts.contains(b))
    .map(|&(a, b)| {
      call(
        "UndirectedEdge",
        vec![vertices[a].clone(), vertices[b].clone()],
      )
    })
    .collect();
  crate::functions::graph::computed_graph_embedding(&[
    Expr::List(vlist.into()),
    Expr::List(elist.into()),
  ])
  .unwrap_or_else(|| {
    // Fall back to a circle.
    let n = verts.len() as f64;
    (0..verts.len())
      .map(|i| {
        let a =
          std::f64::consts::FRAC_PI_2 + i as f64 * std::f64::consts::TAU / n;
        (a.cos(), a.sin())
      })
      .collect()
  })
}

/// `CommunityGraphPlot[g]` and `CommunityGraphPlot[g, {c1, c2, …}]`.
pub fn community_graph_plot_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "CommunityGraphPlot".to_string(),
      args: args.to_vec().into(),
    })
  };
  let Some((vertices, edges)) = args.first().and_then(graph_parts) else {
    return unevaluated();
  };
  let n = vertices.len();
  let is_option =
    |e: &Expr| matches!(e, Expr::Rule { .. } | Expr::RuleDelayed { .. });
  let (given, options): (Option<&Expr>, &[Expr]) = match args.get(1) {
    Some(e) if !is_option(e) => (Some(e), &args[2..]),
    _ => (None, &args[1..]),
  };
  let key = crate::syntax::expr_to_string;
  let index_of = |v: &Expr| vertices.iter().position(|w| key(w) == key(v));
  let mut communities: Vec<Vec<usize>> = match given {
    Some(Expr::List(cs)) => {
      let mut out = Vec::new();
      for c in cs {
        let Expr::List(vs) = c else {
          return unevaluated();
        };
        let mut idx = Vec::new();
        for v in vs {
          match index_of(v) {
            Some(i) if !idx.contains(&i) => idx.push(i),
            Some(_) => {}
            None => return unevaluated(),
          }
        }
        if !idx.is_empty() {
          out.push(idx);
        }
      }
      out
    }
    Some(_) => return unevaluated(),
    None => modularity_communities(n, &edges),
  };
  // Vertices no community names stand on their own.
  for v in 0..n {
    if !communities.iter().any(|c| c.contains(&v)) {
      communities.push(vec![v]);
    }
  }
  let k = communities.len();
  let community_of = |v: usize| communities.iter().position(|c| c.contains(&v));

  // Each community laid out on its own, centred and scaled so its edges
  // are about 0.7 long (as Wolfram draws them).
  let locals: Vec<Vec<(f64, f64)>> = communities
    .iter()
    .map(|c| {
      let pts = sub_layout(&vertices, c, &edges);
      let inner: Vec<f64> = edges
        .iter()
        .filter_map(|&(a, b)| {
          let i = c.iter().position(|&v| v == a)?;
          let j = c.iter().position(|&v| v == b)?;
          let (p, q) = (pts[i], pts[j]);
          Some(((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)).sqrt())
        })
        .filter(|d| *d > 1e-12)
        .collect();
      let scale = if inner.is_empty() {
        1.0
      } else {
        0.7 / (inner.iter().sum::<f64>() / inner.len() as f64)
      };
      let m = pts.len() as f64;
      let cx = pts.iter().map(|p| p.0).sum::<f64>() / m;
      let cy = pts.iter().map(|p| p.1).sum::<f64>() / m;
      pts
        .iter()
        .map(|p| ((p.0 - cx) * scale, (p.1 - cy) * scale))
        .collect()
    })
    .collect();
  let radius_of = |pts: &[(f64, f64)]| {
    pts
      .iter()
      .map(|p| (p.0 * p.0 + p.1 * p.1).sqrt())
      .fold(0.0, f64::max)
  };
  let biggest = locals.iter().map(|l| radius_of(l)).fold(0.0, f64::max);

  // The communities themselves placed as the vertices of the graph of
  // communities (one edge wherever an edge joins two of them), far enough
  // apart that the rings never touch.
  let mut links: Vec<(usize, usize)> = Vec::new();
  for &(a, b) in &edges {
    if let (Some(ca), Some(cb)) = (community_of(a), community_of(b))
      && ca != cb
      && !links.contains(&(ca.min(cb), ca.max(cb)))
    {
      links.push((ca.min(cb), ca.max(cb)));
    }
  }
  let centres: Vec<(f64, f64)> = if k == 1 {
    vec![(0.0, 0.0)]
  } else {
    let ids: Vec<Expr> = (1..=k).map(|i| Expr::Integer(i as i128)).collect();
    let pts = sub_layout(&ids, &(0..k).collect::<Vec<_>>(), &links);
    let mut closest = f64::INFINITY;
    for i in 0..k {
      for j in i + 1..k {
        let d = ((pts[i].0 - pts[j].0).powi(2) + (pts[i].1 - pts[j].1).powi(2))
          .sqrt();
        if d > 1e-12 {
          closest = closest.min(d);
        }
      }
    }
    let spacing = 2.6_f64.max(2.0 * biggest + 1.2);
    let s = if closest.is_finite() {
      spacing / closest
    } else {
      1.0
    };
    pts.iter().map(|p| (p.0 * s, p.1 * s)).collect()
  };
  // Each community is turned so the vertices with edges to another
  // community face it: the angle that maximizes the sum, over those
  // edges, of the vertex's offset along the direction to the other
  // community's centre.
  let mut pos = vec![(0.0, 0.0); n];
  for (c, community) in communities.iter().enumerate() {
    let (mut along, mut across) = (0.0, 0.0);
    for &(a, b) in &edges {
      for (v, u) in [(a, b), (b, a)] {
        let (Some(i), Some(d)) =
          (community.iter().position(|&w| w == v), community_of(u))
        else {
          continue;
        };
        if d == c {
          continue;
        }
        let p = locals[c][i];
        let dir = (centres[d].0 - centres[c].0, centres[d].1 - centres[c].1);
        along += p.0 * dir.0 + p.1 * dir.1;
        across += p.0 * dir.1 - p.1 * dir.0;
      }
    }
    let angle = if along.abs() + across.abs() > 1e-12 {
      across.atan2(along)
    } else {
      0.0
    };
    let (s, co) = angle.sin_cos();
    for (i, &v) in community.iter().enumerate() {
      let (x, y) = locals[c][i];
      pos[v] = (centres[c].0 + co * x - s * y, centres[c].1 + s * x + co * y);
    }
  }
  let extent = {
    let span = |f: &dyn Fn(&(f64, f64)) -> f64| {
      let lo = pos.iter().map(f).fold(f64::INFINITY, f64::min);
      let hi = pos.iter().map(f).fold(f64::NEG_INFINITY, f64::max);
      hi - lo
    };
    span(&|p| p.0).max(span(&|p| p.1))
  };
  let r = crate::functions::graph::wl_vertex_radius(extent.max(1.0));

  let option = |name: &str| {
    options.iter().find_map(|o| match o {
      Expr::Rule {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(s) if s == name) => {
        Some(replacement.as_ref().clone())
      }
      _ => None,
    })
  };
  let boundary = option("CommunityBoundaryStyle")
    .unwrap_or_else(|| call1("Directive", call1("GrayLevel", Expr::Real(0.5))));
  let region = option("CommunityRegionStyle");

  // The rings: a closed B-spline round the vertices' squares of half-side
  // three vertex radii.
  let mut primitives: Vec<Expr> = Vec::new();
  for community in &communities {
    let corners: Vec<(f64, f64, usize)> = community
      .iter()
      .flat_map(|&v| {
        let (x, y) = pos[v];
        let d = 3.0 * r;
        [
          (x - d, y - d),
          (x + d, y - d),
          (x + d, y + d),
          (x - d, y + d),
        ]
      })
      .enumerate()
      .map(|(i, (x, y))| (x, y, i))
      .collect();
    let hull = crate::functions::convex_hull::convex_hull_ccw(&corners);
    let ring: Vec<Expr> = hull
      .iter()
      .map(|&i| point((corners[i].0, corners[i].1)))
      .collect();
    primitives.push(Expr::List(
      vec![
        call1("EdgeForm", boundary.clone()),
        match &region {
          Some(style) => call1("FaceForm", style.clone()),
          None => call("FaceForm", vec![]),
        },
        call1(
          "FilledCurve",
          call(
            "BSplineCurve",
            vec![
              Expr::List(ring.into()),
              rule_expr(id_expr("SplineClosed"), id_expr("True")),
            ],
          ),
        ),
      ]
      .into(),
    ));
  }

  // The edges, in the plain graph style.
  let mut edge_group = vec![call(
    "Directive",
    vec![
      call1("Opacity", Expr::Real(0.7)),
      call(
        "Hue",
        vec![Expr::Real(0.6), Expr::Real(0.7), Expr::Real(0.7)],
      ),
      call1("CapForm", str_expr("Round")),
    ],
  )];
  for &(a, b) in &edges {
    if a != b {
      edge_group.push(call1(
        "Line",
        Expr::List(vec![point(pos[a]), point(pos[b])].into()),
      ));
    }
  }
  primitives.push(Expr::List(edge_group.into()));

  // The vertices, filled and outlined in their community's colour.
  for (c, community) in communities.iter().enumerate() {
    let color = community_color(c);
    let mut group = vec![
      color.clone(),
      call1(
        "EdgeForm",
        Expr::List(vec![color, call1("Opacity", Expr::Integer(1))].into()),
      ),
    ];
    for &v in community {
      group.push(call("Disk", vec![point(pos[v]), Expr::Real(r)]));
    }
    primitives.push(Expr::List(group.into()));
  }

  const PLOT_ONLY: &[&str] =
    &["CommunityBoundaryStyle", "CommunityRegionStyle"];
  let mut graphics_args = vec![Expr::List(primitives.into())];
  graphics_args.extend(options.iter().filter(|o| {
    !matches!(o, Expr::Rule { pattern, .. }
      if matches!(pattern.as_ref(), Expr::Identifier(s) if PLOT_ONLY.contains(&s.as_str())))
  }).cloned());
  graphics_args
    .push(rule_expr(id_expr("FormatType"), id_expr("TraditionalForm")));
  graphics_args.push(rule_expr(id_expr("FrameTicks"), id_expr("None")));
  crate::evaluator::evaluate_function_call_ast("Graphics", &graphics_args)
}
