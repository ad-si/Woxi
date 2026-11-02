//! The legacy `HierarchicalClustering`` package: `Agglomerate`, which builds
//! the merge tree as nested `Cluster[c1, c2, d, n1, n2]` objects,
//! `ClusterFlatten`, which reads its leaves back in order, and
//! `DendrogramPlot`, which draws it.
//!
//! Dissimilarities are kept exact: the pairwise distances are evaluated
//! `DistanceFunction` values (default `SquaredEuclideanDistance`), and a
//! merged cluster's distances follow the Lance–Williams update of the
//! `Linkage` (default `"Single"`), compared numerically to pick each merge.
//! As in the package, `"Average"` weighs the two merged clusters equally and
//! `"WeightedAverage"` by their sizes.

use crate::InterpreterError;
use crate::functions::math_ast::try_eval_to_f64;
use crate::helpers::{call, unevaluated};
use crate::syntax::Expr;

const CTX: &str = "HierarchicalClustering`";

#[allow(clippy::needless_pass_by_value)]
fn ev(e: Expr) -> Result<Expr, InterpreterError> {
  crate::evaluator::evaluate_expr_to_expr(&e)
}

fn option<'a>(opts: &'a [Expr], name: &str) -> Option<&'a Expr> {
  opts.iter().find_map(|o| match o {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == name) => {
      Some(replacement.as_ref())
    }
    _ => None,
  })
}

#[derive(Clone, Copy)]
enum Linkage {
  Single,
  Complete,
  Average,
  WeightedAverage,
  Centroid,
  Median,
  Ward,
}

fn linkage(spec: Option<&Expr>) -> Option<Linkage> {
  Some(match spec {
    None => Linkage::Single,
    Some(Expr::Identifier(a)) if a == "Automatic" => Linkage::Single,
    Some(Expr::String(s)) => match s.as_str() {
      "Single" => Linkage::Single,
      "Complete" => Linkage::Complete,
      "Average" => Linkage::Average,
      "WeightedAverage" => Linkage::WeightedAverage,
      "Centroid" => Linkage::Centroid,
      "Median" => Linkage::Median,
      "Ward" => Linkage::Ward,
      _ => return None,
    },
    _ => return None,
  })
}

/// One node of the merge tree: a leaf (an index into the data) or a merge.
#[derive(Clone)]
enum Tree {
  Leaf(usize),
  Merge {
    left: Box<Tree>,
    right: Box<Tree>,
    distance: Expr,
    sizes: (usize, usize),
  },
}

impl Tree {
  fn size(&self) -> usize {
    match self {
      Tree::Leaf(_) => 1,
      Tree::Merge { sizes, .. } => sizes.0 + sizes.1,
    }
  }
  fn first_index(&self) -> usize {
    match self {
      Tree::Leaf(i) => *i,
      Tree::Merge { left, .. } => left.first_index(),
    }
  }
  fn to_expr(&self, labels: &[Expr]) -> Expr {
    match self {
      Tree::Leaf(i) => labels[*i].clone(),
      Tree::Merge {
        left,
        right,
        distance,
        sizes,
      } => call(
        "Cluster",
        vec![
          left.to_expr(labels),
          right.to_expr(labels),
          distance.clone(),
          Expr::Integer(sizes.0 as i128),
          Expr::Integer(sizes.1 as i128),
        ],
      ),
    }
  }
}

/// `Agglomerate[data]`, `Agglomerate[data -> labels, opts]`.
pub fn agglomerate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || Ok(unevaluated(&format!("{CTX}Agglomerate"), args));
  let Some(first) = args.first() else {
    return original();
  };
  let (data, labels) = match first {
    Expr::Rule {
      pattern,
      replacement,
    } => match (pattern.as_ref(), replacement.as_ref()) {
      (Expr::List(d), Expr::List(l)) if d.len() == l.len() => {
        (d.to_vec(), l.to_vec())
      }
      _ => return original(),
    },
    Expr::List(d) => (d.to_vec(), d.to_vec()),
    _ => return original(),
  };
  let opts = &args[1..];
  let Some(link) = linkage(option(opts, "Linkage")) else {
    return original();
  };
  if data.len() < 2 {
    return original();
  }
  let distance_fn = match option(opts, "DistanceFunction") {
    None => Expr::Identifier("SquaredEuclideanDistance".into()),
    Some(Expr::Identifier(a)) if a == "Automatic" => {
      Expr::Identifier("SquaredEuclideanDistance".into())
    }
    Some(f) => f.clone(),
  };
  let n = data.len();
  let apply = |a: &Expr, b: &Expr| -> Result<Expr, InterpreterError> {
    ev(match &distance_fn {
      Expr::Identifier(name) => call(name, vec![a.clone(), b.clone()]),
      f => Expr::CurriedCall {
        func: Box::new(f.clone()),
        args: vec![a.clone(), b.clone()],
      },
    })
  };
  // Exact and numeric distances between the current clusters.
  let mut dist: Vec<Vec<Expr>> = vec![vec![Expr::Integer(0); n]; n];
  let mut num: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
  for i in 0..n {
    for j in (i + 1)..n {
      let d = apply(&data[i], &data[j])?;
      let Some(v) = try_eval_to_f64(&d) else {
        return original();
      };
      dist[i][j] = d.clone();
      dist[j][i] = d;
      num[i][j] = v;
      num[j][i] = v;
    }
  }
  // Ward's dissimilarity is the growth of the within-cluster sum of
  // squares, half the squared distance for two single points.
  if matches!(link, Linkage::Ward) {
    for i in 0..n {
      for j in 0..n {
        if i != j {
          dist[i][j] = ev(call(
            "Times",
            vec![
              call("Rational", vec![Expr::Integer(1), Expr::Integer(2)]),
              dist[i][j].clone(),
            ],
          ))?;
          num[i][j] /= 2.0;
        }
      }
    }
  }
  let mut clusters: Vec<Option<Tree>> =
    (0..n).map(|i| Some(Tree::Leaf(i))).collect();
  let int = |k: usize| Expr::Integer(k as i128);
  let rat = |p: usize, q: usize| {
    ev(call("Rational", vec![int(p), int(q)])).unwrap_or(Expr::Integer(0))
  };
  for _ in 1..n {
    let active: Vec<usize> =
      (0..n).filter(|&i| clusters[i].is_some()).collect();
    let (mut bi, mut bj, mut best) = (0, 0, f64::INFINITY);
    for (pi, &i) in active.iter().enumerate() {
      for &j in &active[pi + 1..] {
        if num[i][j] < best {
          best = num[i][j];
          bi = i;
          bj = j;
        }
      }
    }
    let ti = clusters[bi].take().unwrap();
    let tj = clusters[bj].take().unwrap();
    let (ni, nj) = (ti.size(), tj.size());
    // The new cluster's distances to the others, by Lance–Williams.
    for &m in &active {
      if m == bi || m == bj {
        continue;
      }
      let nk = clusters[m].as_ref().unwrap().size();
      let (dik, djk, dij) = (
        dist[bi][m].clone(),
        dist[bj][m].clone(),
        dist[bi][bj].clone(),
      );
      let combined = match link {
        Linkage::Single | Linkage::Complete => {
          let smaller = num[bi][m] <= num[bj][m];
          let pick_i = matches!(link, Linkage::Single) == smaller;
          if pick_i { dik } else { djk }
        }
        Linkage::WeightedAverage => ev(call(
          "Plus",
          vec![
            call("Times", vec![rat(ni, ni + nj), dik]),
            call("Times", vec![rat(nj, ni + nj), djk]),
          ],
        ))?,
        Linkage::Average => {
          ev(call("Times", vec![rat(1, 2), call("Plus", vec![dik, djk])]))?
        }
        Linkage::Centroid => ev(call(
          "Plus",
          vec![
            call("Times", vec![rat(ni, ni + nj), dik]),
            call("Times", vec![rat(nj, ni + nj), djk]),
            call(
              "Times",
              vec![Expr::Integer(-1), rat(ni * nj, (ni + nj) * (ni + nj)), dij],
            ),
          ],
        ))?,
        Linkage::Median => ev(call(
          "Plus",
          vec![
            call("Times", vec![rat(1, 2), call("Plus", vec![dik, djk])]),
            call("Times", vec![rat(1, 4), Expr::Integer(-1), dij]),
          ],
        ))?,
        Linkage::Ward => ev(call(
          "Plus",
          vec![
            call("Times", vec![rat(ni + nk, ni + nj + nk), dik]),
            call("Times", vec![rat(nj + nk, ni + nj + nk), djk]),
            call("Times", vec![Expr::Integer(-1), rat(nk, ni + nj + nk), dij]),
          ],
        ))?,
      };
      let v = try_eval_to_f64(&combined).unwrap_or(f64::NAN);
      dist[bi][m] = combined.clone();
      dist[m][bi] = combined;
      num[bi][m] = v;
      num[m][bi] = v;
    }
    let distance = dist[bi][bj].clone();
    let (left, right) = if ti.first_index() <= tj.first_index() {
      (ti, tj)
    } else {
      (tj, ti)
    };
    let sizes = (left.size(), right.size());
    clusters[bi] = Some(Tree::Merge {
      left: Box::new(left),
      right: Box::new(right),
      distance,
      sizes,
    });
  }
  let root = clusters.into_iter().flatten().next().unwrap();
  Ok(root.to_expr(&labels))
}

/// The leaves of a `Cluster` tree, left to right.
fn leaves(e: &Expr, out: &mut Vec<Expr>) {
  match e {
    Expr::FunctionCall { name, args }
      if name == "Cluster" && args.len() == 5 =>
    {
      leaves(&args[0], out);
      leaves(&args[1], out);
    }
    other => out.push(other.clone()),
  }
}

fn is_cluster(e: &Expr) -> bool {
  matches!(e, Expr::FunctionCall { name, args } if name == "Cluster" && args.len() == 5)
}

/// `ClusterFlatten[c]`.
pub fn cluster_flatten_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match args {
    [c] if is_cluster(c) => {
      let mut out = Vec::new();
      leaves(c, &mut out);
      Ok(Expr::List(out.into()))
    }
    _ => Ok(unevaluated(&format!("{CTX}ClusterFlatten"), args)),
  }
}

/// `DendrogramPlot[c, opts]` (or of data, clustered first): one
/// `Line[{{x1, h1}, {x1, h}, {x2, h}, {x2, h2}}]` per merge, leaves at 1, 2,
/// … in `ClusterFlatten` order, with the root pointing up (`Orientation ->
/// Top`), down, left or right.
pub fn dendrogram_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || Ok(unevaluated(&format!("{CTX}DendrogramPlot"), args));
  let Some(first) = args.first() else {
    return original();
  };
  let tree = if is_cluster(first) {
    first.clone()
  } else {
    let mut agg_args = vec![first.clone()];
    agg_args.extend(
      args[1..]
        .iter()
        .filter(|o| {
          option(std::slice::from_ref(*o), "Linkage").is_some()
            || option(std::slice::from_ref(*o), "DistanceFunction").is_some()
        })
        .cloned(),
    );
    let t = agglomerate_ast(&agg_args)?;
    if !is_cluster(&t) {
      return original();
    }
    t
  };
  let opts = &args[1..];
  let orientation = match option(opts, "Orientation") {
    Some(Expr::Identifier(o)) => o.clone(),
    _ => "Top".to_string(),
  };
  let mut lines: Vec<Expr> = Vec::new();
  let mut next_leaf = 1i128;
  // Returns (x position, height) of the node.
  fn walk(
    e: &Expr,
    next_leaf: &mut i128,
    lines: &mut Vec<Expr>,
    place: &dyn Fn(Expr, Expr) -> Expr,
  ) -> Result<(Expr, Expr), InterpreterError> {
    match e {
      Expr::FunctionCall { name, args }
        if name == "Cluster" && args.len() == 5 =>
      {
        let (x1, h1) = walk(&args[0], next_leaf, lines, place)?;
        let (x2, h2) = walk(&args[1], next_leaf, lines, place)?;
        let h = args[2].clone();
        lines.push(call(
          "Line",
          vec![Expr::List(
            vec![
              place(x1.clone(), h1),
              place(x1.clone(), h.clone()),
              place(x2.clone(), h.clone()),
              place(x2.clone(), h2),
            ]
            .into(),
          )],
        ));
        let mid = match (&x1, &x2) {
          (Expr::Integer(a), Expr::Integer(b)) if (a + b) % 2 == 0 => {
            Expr::Integer((a + b) / 2)
          }
          _ => Expr::Real(f64::midpoint(
            try_eval_to_f64(&x1).unwrap_or(0.0),
            try_eval_to_f64(&x2).unwrap_or(0.0),
          )),
        };
        Ok((mid, h))
      }
      _ => {
        let x = Expr::Integer(*next_leaf);
        *next_leaf += 1;
        Ok((x, Expr::Integer(0)))
      }
    }
  }
  let neg = |e: Expr| {
    ev(call("Times", vec![Expr::Integer(-1), e])).unwrap_or(Expr::Integer(0))
  };
  let place = |x: Expr, h: Expr| -> Expr {
    let pair = match orientation.as_str() {
      "Bottom" => vec![x, neg(h)],
      "Left" => vec![neg(h), x],
      "Right" => vec![h, x],
      _ => vec![x, h],
    };
    Expr::List(pair.into())
  };
  walk(&tree, &mut next_leaf, &mut lines, &place)?;
  // Leaf labels, written next to the leaves.
  let mut texts = Vec::new();
  if let Some(f) = option(opts, "LeafLabels")
    && !matches!(f, Expr::Identifier(n) if n == "None")
  {
    let mut leaf_list = Vec::new();
    leaves(&tree, &mut leaf_list);
    for (k, leaf) in leaf_list.iter().enumerate() {
      let label = match f {
        Expr::Identifier(n) if n == "Automatic" || n == "True" => leaf.clone(),
        g => ev(Expr::CurriedCall {
          func: Box::new(g.clone()),
          args: vec![leaf.clone()],
        })?,
      };
      let pos = Expr::Integer(k as i128 + 1);
      let zero = Expr::Integer(0);
      let (offset, at, align) = match orientation.as_str() {
        "Left" => (vec![4, 0], vec![zero, pos], vec![-1, 0]),
        "Right" => (vec![-4, 0], vec![zero, pos], vec![1, 0]),
        "Bottom" => (vec![0, 4], vec![pos, zero], vec![0, -1]),
        _ => (vec![0, -4], vec![pos, zero], vec![0, 1]),
      };
      let ints =
        |v: Vec<i128>| Expr::List(v.into_iter().map(Expr::Integer).collect());
      texts.push(call(
        "Text",
        vec![
          label,
          call("Offset", vec![ints(offset), Expr::List(at.into())]),
          ints(align),
        ],
      ));
    }
  }
  let primitives = Expr::List(
    vec![
      Expr::List(
        vec![
          call(
            "RGBColor",
            vec![Expr::Integer(0), Expr::Integer(1), Expr::Integer(0)],
          ),
          Expr::List(vec![].into()),
        ]
        .into(),
      ),
      Expr::List(
        vec![
          Expr::List(vec![].into()),
          Expr::List(vec![Expr::List(lines.into())].into()),
        ]
        .into(),
      ),
      Expr::List(texts.into()),
    ]
    .into(),
  );
  let mut graphics_args = vec![primitives];
  for o in opts {
    if let Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. } = o
      && let Expr::Identifier(n) = pattern.as_ref()
      && matches!(
        n.as_str(),
        "DistanceFunction"
          | "LeafLabels"
          | "TruncateDendrogram"
          | "HighlightLevel"
          | "HighlightStyle"
          | "Orientation"
          | "PlotStyle"
          | "Linkage"
      )
    {
      continue;
    }
    graphics_args.push(o.clone());
  }
  graphics_args.push(Expr::Rule {
    pattern: Box::new(Expr::Identifier("PlotRange".into())),
    replacement: Box::new(Expr::Identifier("All".into())),
  });
  if option(opts, "AspectRatio").is_none() {
    graphics_args.push(Expr::Rule {
      pattern: Box::new(Expr::Identifier("AspectRatio".into())),
      replacement: Box::new(call(
        "Power",
        vec![Expr::Identifier("GoldenRatio".into()), Expr::Integer(-1)],
      )),
    });
  }
  ev(call("Graphics", graphics_args))
}
