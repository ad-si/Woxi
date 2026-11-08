#[allow(unused_imports)]
use super::*;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::list_helpers_ast::apply_func_to_two_args;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{
  DEFAULT_HEIGHT, DEFAULT_WIDTH, RESOLUTION_SCALE, html_escape,
  parse_image_size,
};

/// Which side of the plot the root of the dendrogram points to.
#[derive(Clone, Copy, PartialEq)]
enum Orientation {
  Top,
  Bottom,
  Left,
  Right,
}

/// Linkage method used to measure the dissimilarity between two clusters
/// (the `ClusterDissimilarityFunction` option).
#[derive(Clone, Copy, PartialEq)]
enum Linkage {
  Single,
  Complete,
  Average,
  WeightedAverage,
  Centroid,
  Median,
  Ward,
}

impl Linkage {
  /// Centroid, Median and Ward use the Lance-Williams update on *squared*
  /// distances; the merge height is the square root of the matrix entry.
  fn uses_squared_distances(self) -> bool {
    matches!(self, Self::Centroid | Self::Median | Self::Ward)
  }
}

/// A node of the binary merge tree. The first `n` nodes are the leaves
/// (in input order); every later node is a merge of two earlier ones.
struct Node {
  /// Child node ids for internal nodes; `None` for leaves.
  children: Option<(usize, usize)>,
  /// Dissimilarity at which the children were merged (0 for leaves).
  height: f64,
  /// Number of leaves under this node.
  size: usize,
}

/// `Dendrogram[data]` — hierarchically cluster `data` and render the
/// resulting merge tree as a Graphics object.
pub fn dendrogram_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("Dendrogram", args));

  let mut orientation = Orientation::Top;
  // `Options[Dendrogram]` has `ClusterDissimilarityFunction -> Single`.
  let mut linkage = Linkage::Single;
  let mut distance_fn: Option<Expr> = None;
  let mut image_width: Option<u32> = None;
  let mut full_width = false;

  // Trailing arguments: an optional orientation plus option rules.
  for opt in &args[1..] {
    if let Some((key, value)) = option_rule(opt) {
      match key.as_str() {
        "DistanceFunction" => {
          if !matches!(&value, Expr::Identifier(s) if s == "Automatic") {
            distance_fn = Some(value);
          }
        }
        "ClusterDissimilarityFunction" => {
          match parse_linkage(&value) {
            Some(l) => linkage = l,
            // A spec we cannot interpret (e.g. a pure function):
            // leave the expression unevaluated rather than guessing.
            None => return unevaluated(),
          }
        }
        "ImageSize" => {
          if let Some((w, _, fw)) =
            parse_image_size(&value, DEFAULT_WIDTH, DEFAULT_HEIGHT)
          {
            image_width = Some(w);
            full_width = fw;
          }
        }
        // Recognized Graphics / clustering options that don't change
        // the algorithm are accepted and ignored.
        _ => {}
      }
    } else if let Expr::Identifier(name) = opt {
      match name.as_str() {
        "Top" => orientation = Orientation::Top,
        "Bottom" => orientation = Orientation::Bottom,
        "Left" => orientation = Orientation::Left,
        "Right" => orientation = Orientation::Right,
        _ => return unevaluated(),
      }
    } else {
      return unevaluated();
    }
  }

  let Some((elements, labels)) = parse_data(&args[0]) else {
    return unevaluated();
  };
  if elements.is_empty() {
    return unevaluated();
  }

  let Some(dist) = distance_matrix(&elements, distance_fn.as_ref()) else {
    return unevaluated();
  };

  let nodes = agglomerate(dist, elements.len(), linkage);
  let svg = render_dendrogram_svg(
    &nodes,
    &labels,
    orientation,
    image_width,
    full_width,
  );
  Ok(crate::graphics_result(svg))
}

/// Split an option argument into `(name, value)` if it is a rule with an
/// identifier on the left-hand side.
fn option_rule(opt: &Expr) -> Option<(String, Expr)> {
  match opt {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      if let Expr::Identifier(s) = pattern.as_ref() {
        Some((s.clone(), (**replacement).clone()))
      } else {
        None
      }
    }
    Expr::FunctionCall { name, args }
      if (name == "Rule" || name == "RuleDelayed") && args.len() == 2 =>
    {
      if let Expr::Identifier(s) = &args[0] {
        Some((s.clone(), args[1].clone()))
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Parse a `ClusterDissimilarityFunction` setting.
fn parse_linkage(value: &Expr) -> Option<Linkage> {
  let name = match value {
    Expr::String(s) => s.as_str(),
    Expr::Identifier(s) if s == "Automatic" => return Some(Linkage::Single),
    _ => return None,
  };
  match name {
    "Single" => Some(Linkage::Single),
    "Complete" => Some(Linkage::Complete),
    "Average" => Some(Linkage::Average),
    "WeightedAverage" => Some(Linkage::WeightedAverage),
    "Centroid" => Some(Linkage::Centroid),
    "Median" => Some(Linkage::Median),
    "Ward" => Some(Linkage::Ward),
    _ => None,
  }
}

/// Extract the data elements and their leaf labels from the first argument.
/// Supports plain lists, `{e1 -> v1, …}`, `{e1, …} -> {v1, …}` and
/// associations `<|v1 -> e1, …|>`.
fn parse_data(arg: &Expr) -> Option<(Vec<Expr>, Vec<String>)> {
  let evaluated = evaluate_expr_to_expr(arg).unwrap_or_else(|_| arg.clone());

  match &evaluated {
    // <|label -> element, …|> — keys label the elements.
    Expr::Association(pairs) => {
      let mut elements = Vec::new();
      let mut labels = Vec::new();
      for (key, val) in pairs {
        elements.push(val.clone());
        labels.push(expr_to_label(key));
      }
      Some((elements, labels))
    }
    // {e1, …} -> {v1, …} — parallel lists of elements and labels.
    Expr::Rule {
      pattern,
      replacement,
    } => {
      if let (Expr::List(es), Expr::List(vs)) =
        (pattern.as_ref(), replacement.as_ref())
        && es.len() == vs.len()
      {
        let labels = vs.iter().map(expr_to_label).collect();
        Some((es.to_vec(), labels))
      } else {
        None
      }
    }
    Expr::List(items) => {
      // {e1 -> v1, …} — each element carries its own label.
      let all_rules = !items.is_empty()
        && items.iter().all(|it| matches!(it, Expr::Rule { .. }));
      if all_rules {
        let mut elements = Vec::new();
        let mut labels = Vec::new();
        for item in items {
          if let Expr::Rule {
            pattern,
            replacement,
          } = item
          {
            elements.push((**pattern).clone());
            labels.push(expr_to_label(replacement));
          }
        }
        return Some((elements, labels));
      }
      // Plain data: label each leaf with the element itself.
      let labels = items.iter().map(expr_to_label).collect();
      Some((items.to_vec(), labels))
    }
    _ => None,
  }
}

/// Render a label expression to display text (strings without their quotes).
fn expr_to_label(expr: &Expr) -> String {
  match expr {
    Expr::String(s) => s.clone(),
    other => expr_to_output(other),
  }
}

/// Compute the pairwise distance matrix for the data elements, using the
/// given `DistanceFunction` or an automatic choice based on the data type.
/// Returns `None` when a distance cannot be established for the data.
pub(crate) fn distance_matrix(
  elements: &[Expr],
  distance_fn: Option<&Expr>,
) -> Option<Vec<Vec<f64>>> {
  let n = elements.len();
  let evaluated: Vec<Expr> = elements
    .iter()
    .map(|e| evaluate_expr_to_expr(e).unwrap_or_else(|_| e.clone()))
    .collect();

  let mut dist = vec![vec![0.0_f64; n]; n];
  let mut fill = |f: &dyn Fn(usize, usize) -> Option<f64>| -> bool {
    for i in 0..n {
      for j in (i + 1)..n {
        match f(i, j) {
          Some(d) if d.is_finite() => {
            dist[i][j] = d;
            dist[j][i] = d;
          }
          _ => return false,
        }
      }
    }
    true
  };

  if let Some(func) = distance_fn {
    let ok = fill(&|i, j| {
      apply_func_to_two_args(func, &evaluated[i], &evaluated[j])
        .ok()
        .and_then(|d| try_eval_to_f64(&d))
    });
    return if ok { Some(dist) } else { None };
  }

  // Automatic: numeric scalars.
  if let Some(nums) = all_scalars(&evaluated) {
    fill(&|i, j| Some((nums[i] - nums[j]).abs()));
    return Some(dist);
  }

  // Automatic: numeric vectors of equal length.
  if let Some(vecs) = all_vectors(&evaluated) {
    fill(&|i, j| {
      let s: f64 = vecs[i]
        .iter()
        .zip(&vecs[j])
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
      Some(s.sqrt())
    });
    return Some(dist);
  }

  // Automatic: strings, compared by edit distance.
  if evaluated.iter().all(|e| matches!(e, Expr::String(_))) {
    let edit = id_expr("EditDistance");
    let ok = fill(&|i, j| {
      apply_func_to_two_args(&edit, &evaluated[i], &evaluated[j])
        .ok()
        .and_then(|d| try_eval_to_f64(&d))
    });
    return if ok { Some(dist) } else { None };
  }

  None
}

/// Try to interpret every element as a real number.
fn all_scalars(elements: &[Expr]) -> Option<Vec<f64>> {
  elements.iter().map(try_eval_to_f64).collect()
}

/// Try to interpret every element as a numeric vector of the same length.
fn all_vectors(elements: &[Expr]) -> Option<Vec<Vec<f64>>> {
  let vecs: Option<Vec<Vec<f64>>> = elements
    .iter()
    .map(|e| {
      if let Expr::List(items) = e {
        items.iter().map(try_eval_to_f64).collect()
      } else {
        None
      }
    })
    .collect();
  let vecs = vecs?;
  let len = vecs.first()?.len();
  if len > 0 && vecs.iter().all(|v| v.len() == len) {
    Some(vecs)
  } else {
    None
  }
}

/// Agglomerative hierarchical clustering via the Lance-Williams update.
/// Returns the full merge tree; the last node is the root.
fn agglomerate(
  mut dist: Vec<Vec<f64>>,
  n: usize,
  linkage: Linkage,
) -> Vec<Node> {
  let total = 2 * n - 1;
  let mut nodes: Vec<Node> = (0..n)
    .map(|_| Node {
      children: None,
      height: 0.0,
      size: 1,
    })
    .collect();

  // Grow the matrix to hold distances for the merged clusters too.
  for row in &mut dist {
    row.resize(total, 0.0);
  }
  dist.resize(total, vec![0.0; total]);

  if linkage.uses_squared_distances() {
    for row in &mut dist {
      for d in row.iter_mut() {
        *d *= *d;
      }
    }
  }

  // The active clusters are kept in a slot list that WL's leaf order
  // depends on: a merge stores the new cluster in the first slot of the
  // pair and moves the last slot into the second one.
  let mut active: Vec<usize> = (0..n).collect();
  while active.len() > 1 {
    // Pick the pair of active clusters with minimal dissimilarity,
    // scanning slot pairs row by row; ties resolve to the last pair found.
    let (mut si, mut sj) = (0, 1);
    let mut best = f64::INFINITY;
    for pi in 0..active.len() {
      for pj in (pi + 1)..active.len() {
        let d = dist[active[pi]][active[pj]];
        if d <= best {
          best = d;
          si = pi;
          sj = pj;
        }
      }
    }
    let (bi, bj) = (active[si], active[sj]);

    let (ni, nj) = (nodes[bi].size as f64, nodes[bj].size as f64);
    let k = nodes.len();
    nodes.push(Node {
      children: Some((bi, bj)),
      height: if linkage.uses_squared_distances() {
        best.max(0.0).sqrt()
      } else {
        best
      },
      size: nodes[bi].size + nodes[bj].size,
    });

    // Lance-Williams update: distance from the new cluster to the rest.
    for &m in &active {
      if m == bi || m == bj {
        continue;
      }
      let nk = nodes[m].size as f64;
      let (ai, aj, beta, gamma) = match linkage {
        Linkage::Single => (0.5, 0.5, 0.0, -0.5),
        Linkage::Complete => (0.5, 0.5, 0.0, 0.5),
        Linkage::Average => (ni / (ni + nj), nj / (ni + nj), 0.0, 0.0),
        Linkage::WeightedAverage => (0.5, 0.5, 0.0, 0.0),
        Linkage::Centroid => (
          ni / (ni + nj),
          nj / (ni + nj),
          -(ni * nj) / ((ni + nj) * (ni + nj)),
          0.0,
        ),
        Linkage::Median => (0.5, 0.5, -0.25, 0.0),
        Linkage::Ward => (
          (ni + nk) / (ni + nj + nk),
          (nj + nk) / (ni + nj + nk),
          -nk / (ni + nj + nk),
          0.0,
        ),
      };
      let d = ai * dist[bi][m]
        + aj * dist[bj][m]
        + beta * dist[bi][bj]
        + gamma * (dist[bi][m] - dist[bj][m]).abs();
      dist[k][m] = d;
      dist[m][k] = d;
    }

    active[si] = k;
    active.swap_remove(sj);
  }

  nodes
}

/// Collect the leaf ids under `node` in drawing order (left to right).
fn leaf_order(nodes: &[Node], node: usize, out: &mut Vec<usize>) {
  match nodes[node].children {
    Some((l, r)) => {
      leaf_order(nodes, l, out);
      leaf_order(nodes, r, out);
    }
    None => out.push(node),
  }
}

/// Estimate the pixel width of a text label at the given font size.
fn est_text_width(label: &str, font_size: f64) -> f64 {
  label.chars().count() as f64 * font_size * 0.6
}

/// Theme colors: `(background, line, label)`. WL draws the tree in
/// `GrayLevel[0.65]` with black leaf labels.
fn theme_colors() -> (&'static str, &'static str, &'static str) {
  if crate::is_dark_mode() {
    ("#1a1a1a", "#bbbbbb", "#dddddd")
  } else {
    ("#ffffff", "#A6A6A6", "#000000")
  }
}

/// Render the merge tree as an SVG dendrogram, laid out like WL's: an
/// `AspectRatio -> 1/Sqrt[2]` picture 360 pt wide for `Top`/`Bottom`, an
/// `AspectRatio -> 2` picture 432 pt tall for `Left`/`Right`. The first
/// leaf sits at the left (`Top`/`Bottom`) or at the bottom (`Left`/`Right`).
fn render_dendrogram_svg(
  nodes: &[Node],
  labels: &[String],
  orientation: Orientation,
  image_width: Option<u32>,
  full_width: bool,
) -> String {
  let sf = RESOLUTION_SCALE as f64;
  let (bg_color, line_color, label_color) = theme_colors();
  let font_size = 10.0;

  let root = nodes.len() - 1;
  let mut order = Vec::new();
  leaf_order(nodes, root, &mut order);
  let n_leaves = order.len();

  // Abstract coordinates: `u` spreads the leaves across [0, 1] (first leaf
  // at 0) and `v` is the merge height normalized to [0, 1].
  let h_max = nodes
    .iter()
    .map(|nd| nd.height)
    .fold(0.0_f64, f64::max)
    .max(f64::EPSILON);
  let mut u = vec![0.0_f64; nodes.len()];
  for (slot, &leaf) in order.iter().enumerate() {
    u[leaf] = if n_leaves > 1 {
      slot as f64 / (n_leaves - 1) as f64
    } else {
      0.5
    };
  }
  // Internal nodes sit midway between their children (children always
  // have smaller ids than their parent).
  for id in 0..nodes.len() {
    if let Some((l, r)) = nodes[id].children {
      u[id] = f64::midpoint(u[l], u[r]);
    }
  }
  let v: Vec<f64> = nodes.iter().map(|nd| nd.height / h_max).collect();

  let longest_label = labels
    .iter()
    .map(|l| est_text_width(l, font_size))
    .fold(0.0_f64, f64::max);
  let vertical = matches!(orientation, Orientation::Top | Orientation::Bottom);

  // Picture size and the plot box (in pt): `leaf0..leaf1` along the leaf
  // axis, `root..tip` along the height axis.
  let (width, height, leaf0, leaf1, root_at, tip_at);
  if vertical {
    let w = image_width.unwrap_or(DEFAULT_WIDTH) as f64;
    // Measured from WL's 360 pt picture: leaves from 7.18 to 351.82 pt,
    // the root 6.90 pt below the top, the tips 240.05 pt below it and a
    // 15.05 pt label strip under them.
    let root_pad = 0.01917 * w;
    let depth = 0.6668 * w;
    let strip = 15.05;
    width = w;
    height = root_pad + depth + strip;
    (leaf0, leaf1) = (0.01994 * w, w - 0.02272 * w);
    if orientation == Orientation::Top {
      (root_at, tip_at) = (root_pad, root_pad + depth);
    } else {
      (root_at, tip_at) = (height - root_pad, height - root_pad - depth);
    }
  } else {
    let label_room = 1.0 + longest_label + 2.7;
    let h = match image_width {
      Some(w) => 2.0 * w as f64,
      None => 432.0,
    };
    let leaf_pad = 0.01640 * h;
    let root_pad = 0.00986 * h;
    let depth = 0.4734 * h;
    height = h;
    width = (root_pad + depth + label_room).round();
    // The first leaf at the bottom.
    (leaf0, leaf1) = (h - leaf_pad, leaf_pad);
    if orientation == Orientation::Left {
      (root_at, tip_at) = (root_pad, root_pad + depth);
    } else {
      (root_at, tip_at) = (width - root_pad, width - root_pad - depth);
    }
  }

  // Map abstract (u, v) coordinates to render units.
  let to_px = |uu: f64, vv: f64| -> (f64, f64) {
    let along = leaf0 + uu * (leaf1 - leaf0);
    let across = tip_at + vv * (root_at - tip_at);
    if vertical {
      (along * sf, across * sf)
    } else {
      (across * sf, along * sf)
    }
  };

  let svg_width = width.round().max(1.0) as u32;
  let svg_height = height.round().max(1.0) as u32;
  let render_width = svg_width as f64 * sf;
  let render_height = svg_height as f64 * sf;

  let mut body = String::new();
  body.push_str(&format!(
    "<rect width=\"{render_width:.0}\" height=\"{render_height:.0}\" fill=\"{bg_color}\"/>\n"
  ));

  // Draw the elbow connector of every merge.
  let stroke = sf;
  for (id, node) in nodes.iter().enumerate() {
    let Some((l, r)) = node.children else {
      continue;
    };
    let (x1, y1) = to_px(u[l], v[l]);
    let (x1b, y1b) = to_px(u[l], v[id]);
    let (x2b, y2b) = to_px(u[r], v[id]);
    let (x2, y2) = to_px(u[r], v[r]);
    body.push_str(&format!(
      "<path d=\"M {x1:.1} {y1:.1} L {x1b:.1} {y1b:.1} L {x2b:.1} {y2b:.1} L {x2:.1} {y2:.1}\" \
       fill=\"none\" stroke=\"{line_color}\" stroke-width=\"{stroke:.1}\" \
       stroke-linecap=\"square\"/>\n"
    ));
  }

  // Leaf labels next to the leaf tips.
  for &leaf in &order {
    let label = labels.get(leaf).map_or("", String::as_str);
    if label.is_empty() {
      continue;
    }
    let (x, y) = to_px(u[leaf], 0.0);
    let (lx, ly, anchor) = match orientation {
      Orientation::Top => (x, y + 10.05 * sf, "middle"),
      Orientation::Bottom => (x, y - 3.0 * sf, "middle"),
      Orientation::Left => (x + 1.0 * sf, y + 3.46 * sf, "start"),
      Orientation::Right => (x - 1.0 * sf, y + 3.46 * sf, "end"),
    };
    body.push_str(&format!(
      "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"{}\" \
       font-family=\"sans-serif\" font-size=\"{:.0}\" fill=\"{}\">{}</text>\n",
      lx,
      ly,
      anchor,
      font_size * sf,
      label_color,
      html_escape(label)
    ));
  }

  let mut buf = format!(
    "<svg width=\"{svg_width}\" height=\"{svg_height}\" viewBox=\"0 0 {render_width:.0} {render_height:.0}\" \
     xmlns=\"http://www.w3.org/2000/svg\">\n{body}</svg>"
  );

  if full_width {
    let old = format!("width=\"{svg_width}\" height=\"{svg_height}\"");
    buf = buf.replacen(&old, "width=\"100%\"", 1);
  }

  buf
}
