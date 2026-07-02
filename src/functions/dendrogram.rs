use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::list_helpers_ast::apply_func_to_two_args;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{
  DEFAULT_HEIGHT, DEFAULT_WIDTH, RESOLUTION_SCALE, parse_image_size,
};
use crate::syntax::Expr;

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
    matches!(self, Linkage::Centroid | Linkage::Median | Linkage::Ward)
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
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "Dendrogram".to_string(),
      args: args.to_vec().into(),
    })
  };

  let mut orientation = Orientation::Top;
  let mut linkage = Linkage::Ward;
  let mut distance_fn: Option<Expr> = None;
  let mut svg_width = DEFAULT_WIDTH;
  let mut svg_height = DEFAULT_HEIGHT;
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
          if let Some((w, h, fw)) =
            parse_image_size(&value, DEFAULT_WIDTH, DEFAULT_HEIGHT)
          {
            svg_width = w;
            svg_height = h;
            full_width = fw;
          }
        }
        // Recognised Graphics / clustering options that don't change
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
    svg_width,
    svg_height,
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
    Expr::Identifier(s) if s == "Automatic" => return Some(Linkage::Ward),
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
    other => crate::syntax::expr_to_output(other),
  }
}

/// Compute the pairwise distance matrix for the data elements, using the
/// given `DistanceFunction` or an automatic choice based on the data type.
/// Returns `None` when a distance cannot be established for the data.
fn distance_matrix(
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
    let edit = Expr::Identifier("EditDistance".to_string());
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
  for row in dist.iter_mut() {
    row.resize(total, 0.0);
  }
  dist.resize(total, vec![0.0; total]);

  if linkage.uses_squared_distances() {
    for row in dist.iter_mut() {
      for d in row.iter_mut() {
        *d *= *d;
      }
    }
  }

  let mut active: Vec<usize> = (0..n).collect();
  while active.len() > 1 {
    // Pick the pair of active clusters with minimal dissimilarity;
    // ties resolve to the earliest-created pair for determinism.
    let (mut bi, mut bj) = (0, 1);
    let mut best = f64::INFINITY;
    for (pi, &i) in active.iter().enumerate() {
      for &j in &active[(pi + 1)..] {
        if dist[i][j] < best {
          best = dist[i][j];
          bi = i;
          bj = j;
        }
      }
    }

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

    active.retain(|&x| x != bi && x != bj);
    active.push(k);
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

/// Escape special HTML characters in text content.
fn html_escape(s: &str) -> String {
  s.replace('&', "&amp;")
    .replace('<', "&lt;")
    .replace('>', "&gt;")
    .replace('"', "&quot;")
}

/// Theme colors: `(background, line, label)`.
fn theme_colors() -> (&'static str, &'static str, &'static str) {
  if crate::is_dark_mode() {
    ("#1a1a1a", "#bbbbbb", "#999999")
  } else {
    ("#ffffff", "#333333", "#555555")
  }
}

/// Render the merge tree as an SVG dendrogram.
fn render_dendrogram_svg(
  nodes: &[Node],
  labels: &[String],
  orientation: Orientation,
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
) -> String {
  let sf = RESOLUTION_SCALE as f64;
  let render_width = svg_width as f64 * sf;
  let render_height = svg_height as f64 * sf;

  let (bg_color, line_color, label_color) = theme_colors();
  let font_size = 11.0 * sf;
  let label_gap = 4.0 * sf;

  let root = nodes.len() - 1;
  let mut order = Vec::new();
  leaf_order(nodes, root, &mut order);
  let n_leaves = order.len();

  // Abstract coordinates: `u` spreads the leaves across [0, 1] and `v`
  // is the merge height normalised to [0, 1] (leaves at 0, root at 1).
  let h_max = nodes
    .iter()
    .map(|nd| nd.height)
    .fold(0.0_f64, f64::max)
    .max(f64::EPSILON);

  let mut u = vec![0.0_f64; nodes.len()];
  for (slot, &leaf) in order.iter().enumerate() {
    u[leaf] = (slot as f64 + 0.5) / n_leaves as f64;
  }
  // Internal nodes sit midway between their children (post-order walk:
  // children always have smaller ids than their parent).
  for id in 0..nodes.len() {
    if let Some((l, r)) = nodes[id].children {
      u[id] = (u[l] + u[r]) / 2.0;
    }
  }
  let v: Vec<f64> = nodes.iter().map(|nd| nd.height / h_max).collect();

  // Reserve space for the leaf labels on the side where they appear.
  let longest_label = labels
    .iter()
    .map(|l| est_text_width(l, font_size))
    .fold(0.0_f64, f64::max);
  let side_pad = 10.0 * sf;
  let label_strip = font_size * 1.4 + label_gap;

  let (mx0, mx1, my0, my1) = match orientation {
    // Root on top, leaves and labels at the bottom.
    Orientation::Top => (side_pad, side_pad, side_pad, label_strip),
    // Root at the bottom, leaves and labels on top.
    Orientation::Bottom => (side_pad, side_pad, label_strip, side_pad),
    // Root on the left, leaves and labels on the right.
    Orientation::Left => (
      side_pad,
      (longest_label + label_gap + side_pad).min(render_width * 0.5),
      side_pad,
      side_pad,
    ),
    // Root on the right, leaves and labels on the left.
    Orientation::Right => (
      (longest_label + label_gap + side_pad).min(render_width * 0.5),
      side_pad,
      side_pad,
      side_pad,
    ),
  };
  let plot_w = (render_width - mx0 - mx1).max(1.0);
  let plot_h = (render_height - my0 - my1).max(1.0);

  // Map abstract (u, v) coordinates to pixels for the chosen orientation.
  let to_px = |uu: f64, vv: f64| -> (f64, f64) {
    match orientation {
      Orientation::Top => (mx0 + uu * plot_w, my0 + (1.0 - vv) * plot_h),
      Orientation::Bottom => (mx0 + uu * plot_w, my0 + vv * plot_h),
      Orientation::Left => (mx0 + (1.0 - vv) * plot_w, my0 + uu * plot_h),
      Orientation::Right => (mx0 + vv * plot_w, my0 + uu * plot_h),
    }
  };

  let mut body = String::new();
  body.push_str(&format!(
    "<rect width=\"{:.0}\" height=\"{:.0}\" fill=\"{}\"/>\n",
    render_width, render_height, bg_color
  ));

  // Draw the elbow connector of every merge.
  let stroke = 1.2 * sf;
  for (id, node) in nodes.iter().enumerate() {
    let Some((l, r)) = node.children else {
      continue;
    };
    let (x1, y1) = to_px(u[l], v[l]);
    let (x1b, y1b) = to_px(u[l], v[id]);
    let (x2b, y2b) = to_px(u[r], v[id]);
    let (x2, y2) = to_px(u[r], v[r]);
    body.push_str(&format!(
      "<path d=\"M {:.1} {:.1} L {:.1} {:.1} L {:.1} {:.1} L {:.1} {:.1}\" \
       fill=\"none\" stroke=\"{}\" stroke-width=\"{:.1}\" \
       stroke-linecap=\"round\" stroke-linejoin=\"round\"/>\n",
      x1, y1, x1b, y1b, x2b, y2b, x2, y2, line_color, stroke
    ));
  }

  // Leaf labels next to the leaf tips.
  for &leaf in &order {
    let label = labels.get(leaf).map(String::as_str).unwrap_or("");
    if label.is_empty() {
      continue;
    }
    let (x, y) = to_px(u[leaf], 0.0);
    let (lx, ly, anchor, dy) = match orientation {
      Orientation::Top => (x, y + label_gap, "middle", "0.9em"),
      Orientation::Bottom => (x, y - label_gap, "middle", "-0.2em"),
      Orientation::Left => (x + label_gap, y, "start", "0.32em"),
      Orientation::Right => (x - label_gap, y, "end", "0.32em"),
    };
    body.push_str(&format!(
      "<text x=\"{:.1}\" y=\"{:.1}\" dy=\"{}\" text-anchor=\"{}\" \
       font-family=\"sans-serif\" font-size=\"{:.0}\" fill=\"{}\">{}</text>\n",
      lx,
      ly,
      dy,
      anchor,
      font_size,
      label_color,
      html_escape(label)
    ));
  }

  let mut buf = format!(
    "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {:.0} {:.0}\" \
     xmlns=\"http://www.w3.org/2000/svg\">\n{}</svg>",
    svg_width, svg_height, render_width, render_height, body
  );

  if full_width {
    let old = format!("width=\"{}\" height=\"{}\"", svg_width, svg_height);
    buf = buf.replacen(&old, "width=\"100%\"", 1);
  }

  buf
}
