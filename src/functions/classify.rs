//! `Classify`, the `ClassifierFunction` it returns, and
//! `ClassifierMeasurements`.
//!
//! Examples are numbers or numeric vectors; classes are arbitrary
//! expressions. A trained classifier is the inert object
//! `ClassifierFunction[<|"Method" -> m, "Classes" -> {…}, "Examples" -> {…},
//! "Labels" -> {…}, "ModelID" -> id|>]`: the training set travels with it, and
//! the fitted model is kept in a per-thread cache under `"ModelID"` (and
//! refitted from the stored training set when the cache misses), so applying
//! the classifier at many points — e.g. inside `ContourPlot` — does not
//! retrain it each time.
//!
//! Every method trains deterministically: the few places that need random
//! numbers (bootstrap samples, initial network weights) draw them from a
//! fixed-seed generator, so the same data always gives the same classifier.
//!
//! Supported methods: `"LogisticRegression"` (multinomial, L2-regularized),
//! `"NaiveBayes"` (Gaussian per feature), `"NearestNeighbors"`,
//! `"NeuralNetwork"` (one hidden tanh layer, softmax output),
//! `"SupportVectorMachine"` (RBF kernel, one-vs-rest, Platt-scaled
//! probabilities), `"DecisionTree"` and `"RandomForest"` (CART with Gini
//! impurity). `Automatic` picks `"LogisticRegression"`.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::InterpreterError;
use crate::functions::list_helpers_ast::sorting::compare_exprs;
use crate::functions::math_ast::try_eval_to_f64;
use crate::helpers::{call1, unevaluated};
use crate::syntax::{Expr, expr_to_string};

const METHODS: &[&str] = &[
  "DecisionTree",
  "LogisticRegression",
  "NaiveBayes",
  "NearestNeighbors",
  "NeuralNetwork",
  "RandomForest",
  "SupportVectorMachine",
];

// ─── Training data ─────────────────────────────────────────────────────────

/// A training or test set: feature vectors with the index of their class.
struct Dataset {
  x: Vec<Vec<f64>>,
  y: Vec<usize>,
}

/// One example's features: a number is a one-feature vector.
fn features(e: &Expr) -> Option<Vec<f64>> {
  match e {
    Expr::List(items) => items.iter().map(try_eval_to_f64).collect(),
    _ => try_eval_to_f64(e).map(|v| vec![v]),
  }
}

fn rule_parts(e: &Expr) -> Option<(&Expr, &Expr)> {
  match e {
    Expr::Rule {
      pattern,
      replacement,
    } => Some((pattern, replacement)),
    Expr::FunctionCall { name, args } if name == "Rule" && args.len() == 2 => {
      Some((&args[0], &args[1]))
    }
    _ => None,
  }
}

fn is_option(e: &Expr) -> bool {
  matches!(e, Expr::Rule { .. } | Expr::RuleDelayed { .. })
    || matches!(e, Expr::FunctionCall { name, args }
      if (name == "Rule" || name == "RuleDelayed") && args.len() == 2)
}

/// The (example, class) pairs of a training/test specification:
/// `<|c1 -> {x…}, …|>`, `{x1 -> c1, …}` or `{x1, …} -> {c1, …}`.
fn labeled_pairs(spec: &Expr) -> Option<Vec<(Expr, Expr)>> {
  match spec {
    Expr::Association(entries) => {
      let mut out = Vec::new();
      for (class, examples) in entries {
        let Expr::List(xs) = examples else {
          return None;
        };
        out.extend(xs.iter().map(|x| (x.clone(), class.clone())));
      }
      Some(out)
    }
    Expr::List(items) => items
      .iter()
      .map(|r| rule_parts(r).map(|(x, c)| (x.clone(), c.clone())))
      .collect(),
    _ => {
      let (xs, cs) = rule_parts(spec)?;
      match (xs, cs) {
        (Expr::List(xs), Expr::List(cs)) if xs.len() == cs.len() => Some(
          xs.iter()
            .cloned()
            .zip(cs.iter().cloned())
            .collect::<Vec<_>>(),
        ),
        _ => None,
      }
    }
  }
}

fn same_expr(a: &Expr, b: &Expr) -> bool {
  expr_to_string(a) == expr_to_string(b)
}

fn class_index(classes: &[Expr], c: &Expr) -> Option<usize> {
  classes.iter().position(|k| same_expr(k, c))
}

// ─── Models ────────────────────────────────────────────────────────────────

/// Per-feature affine standardization fitted on the training set.
#[derive(Clone)]
struct Scaler {
  mean: Vec<f64>,
  scale: Vec<f64>,
}

impl Scaler {
  fn fit(x: &[Vec<f64>]) -> Self {
    let d = x.first().map_or(0, Vec::len);
    let n = x.len().max(1) as f64;
    let mut mean = vec![0.0; d];
    for row in x {
      for (m, v) in mean.iter_mut().zip(row) {
        *m += v / n;
      }
    }
    let mut var = vec![0.0; d];
    for row in x {
      for ((s, v), m) in var.iter_mut().zip(row).zip(&mean) {
        *s += (v - m) * (v - m) / n;
      }
    }
    let scale = var
      .into_iter()
      .map(|v| if v > 1e-300 { v.sqrt() } else { 1.0 })
      .collect();
    Scaler { mean, scale }
  }

  fn apply(&self, row: &[f64]) -> Vec<f64> {
    row
      .iter()
      .zip(&self.mean)
      .zip(&self.scale)
      .map(|((v, m), s)| (v - m) / s)
      .collect()
  }
}

/// A small deterministic generator (SplitMix64).
struct Rng(u64);

impl Rng {
  fn next_u64(&mut self) -> u64 {
    self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
  }

  fn uniform(&mut self) -> f64 {
    (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
  }

  fn below(&mut self, n: usize) -> usize {
    (self.next_u64() % n as u64) as usize
  }
}

fn softmax(z: &[f64]) -> Vec<f64> {
  let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
  let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
  let s: f64 = e.iter().sum();
  e.into_iter().map(|v| v / s).collect()
}

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
  a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum()
}

/// Class frequencies with a light Laplace smoothing, so no class gets
/// probability exactly 0.
fn smoothed(counts: &[f64], alpha: f64) -> Vec<f64> {
  let total: f64 = counts.iter().sum::<f64>() + alpha * counts.len() as f64;
  counts.iter().map(|c| (c + alpha) / total).collect()
}

enum Model {
  Logistic {
    scaler: Scaler,
    /// One row of weights per class; the last entry is the bias.
    w: Vec<Vec<f64>>,
  },
  NaiveBayes {
    prior: Vec<f64>,
    mean: Vec<Vec<f64>>,
    var: Vec<Vec<f64>>,
  },
  Neighbors {
    scaler: Scaler,
    x: Vec<Vec<f64>>,
    y: Vec<usize>,
    k: usize,
  },
  Network {
    scaler: Scaler,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
  },
  Svm {
    scaler: Scaler,
    x: Vec<Vec<f64>>,
    gamma: f64,
    /// Per one-vs-rest machine: signed dual weights, bias, Platt (A, B).
    machines: Vec<(Vec<f64>, f64, f64, f64)>,
  },
  Forest(Vec<Tree>),
}

enum Tree {
  Leaf(Vec<f64>),
  Split {
    feature: usize,
    threshold: f64,
    left: Box<Tree>,
    right: Box<Tree>,
  },
}

impl Tree {
  fn probabilities(&self, x: &[f64]) -> &[f64] {
    match self {
      Tree::Leaf(p) => p,
      Tree::Split {
        feature,
        threshold,
        left,
        right,
      } => {
        if x[*feature] <= *threshold {
          left.probabilities(x)
        } else {
          right.probabilities(x)
        }
      }
    }
  }
}

fn gini(counts: &[f64], total: f64) -> f64 {
  if total <= 0.0 {
    return 0.0;
  }
  1.0
    - counts
      .iter()
      .map(|c| (c / total) * (c / total))
      .sum::<f64>()
}

/// Grow a CART tree on the rows `idx`, considering `n_try` random features
/// at each split (all of them when `n_try` is the feature count).
fn grow_tree(
  data: &Dataset,
  idx: &[usize],
  n_classes: usize,
  n_try: usize,
  rng: &mut Rng,
  depth: usize,
) -> Tree {
  let mut counts = vec![0.0; n_classes];
  for &i in idx {
    counts[data.y[i]] += 1.0;
  }
  let pure = counts.iter().filter(|&&c| c > 0.0).count() <= 1;
  if pure || idx.len() < 2 || depth >= 32 {
    return Tree::Leaf(smoothed(&counts, 0.05));
  }
  let d = data.x[0].len();
  let mut feats: Vec<usize> = (0..d).collect();
  for i in (1..d).rev() {
    feats.swap(i, rng.below(i + 1));
  }
  let total = idx.len() as f64;
  let parent = gini(&counts, total);
  let mut best: Option<(f64, usize, f64)> = None;
  for &f in feats.iter().take(n_try.max(1)) {
    let mut order = idx.to_vec();
    order.sort_by(|&a, &b| data.x[a][f].total_cmp(&data.x[b][f]));
    let mut left = vec![0.0; n_classes];
    for k in 0..order.len() - 1 {
      left[data.y[order[k]]] += 1.0;
      let (a, b) = (data.x[order[k]][f], data.x[order[k + 1]][f]);
      if a == b {
        continue;
      }
      let nl = (k + 1) as f64;
      let nr = total - nl;
      let right: Vec<f64> =
        counts.iter().zip(&left).map(|(c, l)| c - l).collect();
      let impurity = (nl * gini(&left, nl) + nr * gini(&right, nr)) / total;
      if best.is_none_or(|(g, _, _)| impurity < g - 1e-12) {
        best = Some((impurity, f, f64::midpoint(a, b)));
      }
    }
  }
  match best {
    Some((g, feature, threshold)) if g < parent - 1e-12 => {
      let (l, r): (Vec<usize>, Vec<usize>) =
        idx.iter().partition(|&&i| data.x[i][feature] <= threshold);
      Tree::Split {
        feature,
        threshold,
        left: Box::new(grow_tree(data, &l, n_classes, n_try, rng, depth + 1)),
        right: Box::new(grow_tree(data, &r, n_classes, n_try, rng, depth + 1)),
      }
    }
    _ => Tree::Leaf(smoothed(&counts, 0.05)),
  }
}

fn train_logistic(data: &Dataset, m: usize) -> Model {
  let scaler = Scaler::fit(&data.x);
  let xs: Vec<Vec<f64>> = data.x.iter().map(|r| scaler.apply(r)).collect();
  let d = xs[0].len();
  let n = xs.len() as f64;
  let lambda = 1e-3;
  let mut w = vec![vec![0.0; d + 1]; m];
  let rate = 0.5;
  for _ in 0..2000 {
    let mut grad = vec![vec![0.0; d + 1]; m];
    for (row, &yi) in xs.iter().zip(&data.y) {
      let z: Vec<f64> = w
        .iter()
        .map(|wc| wc[d] + wc.iter().zip(row).map(|(a, b)| a * b).sum::<f64>())
        .collect();
      let p = softmax(&z);
      for c in 0..m {
        let err = p[c] - if c == yi { 1.0 } else { 0.0 };
        for j in 0..d {
          grad[c][j] += err * row[j] / n;
        }
        grad[c][d] += err / n;
      }
    }
    for c in 0..m {
      for j in 0..=d {
        let reg = if j < d { lambda * w[c][j] } else { 0.0 };
        w[c][j] -= rate * (grad[c][j] + reg);
      }
    }
  }
  Model::Logistic { scaler, w }
}

fn train_naive_bayes(data: &Dataset, m: usize) -> Model {
  let d = data.x[0].len();
  let all = Scaler::fit(&data.x);
  let mut counts = vec![0.0_f64; m];
  let mut mean = vec![vec![0.0; d]; m];
  for (row, &c) in data.x.iter().zip(&data.y) {
    counts[c] += 1.0;
    for j in 0..d {
      mean[c][j] += row[j];
    }
  }
  for c in 0..m {
    for v in &mut mean[c] {
      *v /= counts[c].max(1.0);
    }
  }
  let mut var = vec![vec![0.0; d]; m];
  for (row, &c) in data.x.iter().zip(&data.y) {
    for j in 0..d {
      var[c][j] += (row[j] - mean[c][j]).powi(2);
    }
  }
  for c in 0..m {
    for j in 0..d {
      // Shrink toward the overall spread so a class with one example (or
      // constant features) still has a usable width.
      let floor = 1e-2 * all.scale[j] * all.scale[j];
      var[c][j] = var[c][j] / counts[c].max(1.0) + floor;
    }
  }
  Model::NaiveBayes {
    prior: smoothed(&counts, 1.0),
    mean,
    var,
  }
}

fn train_neighbors(data: &Dataset) -> Model {
  let scaler = Scaler::fit(&data.x);
  let x: Vec<Vec<f64>> = data.x.iter().map(|r| scaler.apply(r)).collect();
  let n = x.len();
  // An odd neighborhood growing slowly with the training-set size.
  let mut k = ((n as f64).sqrt() / 2.0).round().max(1.0) as usize;
  if k.is_multiple_of(2) {
    k += 1;
  }
  Model::Neighbors {
    scaler,
    x,
    y: data.y.clone(),
    k: k.min(n),
  }
}

fn train_network(data: &Dataset, m: usize) -> Model {
  let scaler = Scaler::fit(&data.x);
  let xs: Vec<Vec<f64>> = data.x.iter().map(|r| scaler.apply(r)).collect();
  let d = xs[0].len();
  let h = 16;
  let mut rng = Rng(0x5EED_0001);
  let mut init = |fan_in: usize| {
    let s = (1.0 / fan_in as f64).sqrt();
    (rng.uniform() * 2.0 - 1.0) * s * 1.7
  };
  let mut w1: Vec<Vec<f64>> =
    (0..h).map(|_| (0..d).map(|_| init(d)).collect()).collect();
  let mut b1 = vec![0.0; h];
  let mut w2: Vec<Vec<f64>> =
    (0..m).map(|_| (0..h).map(|_| init(h)).collect()).collect();
  let mut b2 = vec![0.0; m];
  let n = xs.len() as f64;
  // Full-batch Adam.
  let (beta1, beta2, eps, rate, lambda) = (0.9, 0.999, 1e-8, 0.02, 1e-4);
  let size = h * d + h + m * h + m;
  let mut mom = vec![0.0; size];
  let mut vel = vec![0.0; size];
  for step in 1..=1500 {
    let mut g = vec![0.0; size];
    for (row, &yi) in xs.iter().zip(&data.y) {
      let hidden: Vec<f64> = (0..h)
        .map(|u| {
          (b1[u] + w1[u].iter().zip(row).map(|(a, b)| a * b).sum::<f64>())
            .tanh()
        })
        .collect();
      let z: Vec<f64> = (0..m)
        .map(|c| {
          b2[c] + w2[c].iter().zip(&hidden).map(|(a, b)| a * b).sum::<f64>()
        })
        .collect();
      let p = softmax(&z);
      let mut dh = vec![0.0; h];
      for c in 0..m {
        let err = (p[c] - if c == yi { 1.0 } else { 0.0 }) / n;
        for u in 0..h {
          g[h * d + h + c * h + u] += err * hidden[u];
          dh[u] += err * w2[c][u];
        }
        g[h * d + h + m * h + c] += err;
      }
      for u in 0..h {
        let dz = dh[u] * (1.0 - hidden[u] * hidden[u]);
        for j in 0..d {
          g[u * d + j] += dz * row[j];
        }
        g[h * d + u] += dz;
      }
    }
    let t = step as f64;
    let mut k = 0;
    let mut update = |param: &mut f64, grad: f64, reg: bool| {
      let grad = grad + if reg { lambda * *param } else { 0.0 };
      mom[k] = beta1 * mom[k] + (1.0 - beta1) * grad;
      vel[k] = beta2 * vel[k] + (1.0 - beta2) * grad * grad;
      let mh = mom[k] / (1.0 - beta1.powf(t));
      let vh = vel[k] / (1.0 - beta2.powf(t));
      *param -= rate * mh / (vh.sqrt() + eps);
      k += 1;
    };
    for u in 0..h {
      for j in 0..d {
        update(&mut w1[u][j], g[u * d + j], true);
      }
    }
    for u in 0..h {
      update(&mut b1[u], g[h * d + u], false);
    }
    for c in 0..m {
      for u in 0..h {
        update(&mut w2[c][u], g[h * d + h + c * h + u], true);
      }
    }
    for c in 0..m {
      update(&mut b2[c], g[h * d + h + m * h + c], false);
    }
  }
  Model::Network {
    scaler,
    w1,
    b1,
    w2,
    b2,
  }
}

fn rbf(a: &[f64], b: &[f64], gamma: f64) -> f64 {
  (-gamma * sq_dist(a, b)).exp()
}

/// Binary soft-margin SVM by SMO with the maximal-violating-pair working
/// set; returns the signed dual weights `alpha_i y_i` and the bias.
fn smo(kernel: &[Vec<f64>], y: &[f64], c: f64) -> (Vec<f64>, f64) {
  let n = y.len();
  let mut alpha = vec![0.0; n];
  // Gradient of the dual objective: G_i = (Q alpha)_i - 1.
  let mut grad = vec![-1.0; n];
  let tol = 1e-3;
  for _ in 0..(20 * n).max(1000) {
    let (mut i, mut gmax) = (usize::MAX, f64::NEG_INFINITY);
    let (mut j, mut gmin) = (usize::MAX, f64::INFINITY);
    for t in 0..n {
      let v = -y[t] * grad[t];
      let up = (y[t] > 0.0 && alpha[t] < c) || (y[t] < 0.0 && alpha[t] > 0.0);
      let low = (y[t] > 0.0 && alpha[t] > 0.0) || (y[t] < 0.0 && alpha[t] < c);
      if up && v > gmax {
        gmax = v;
        i = t;
      }
      if low && v < gmin {
        gmin = v;
        j = t;
      }
    }
    if i == usize::MAX || j == usize::MAX || gmax - gmin < tol {
      break;
    }
    let eta = (kernel[i][i] + kernel[j][j] - 2.0 * kernel[i][j]).max(1e-12);
    // Move along y_i e_i - y_j e_j, clipped to the box.
    let mut step = (gmax - gmin) / eta;
    let room = |t: usize, dir: f64| {
      // dir: sign of the change in alpha_t.
      if dir > 0.0 { c - alpha[t] } else { alpha[t] }
    };
    step = step.min(room(i, y[i])).min(room(j, -y[j]));
    let (di, dj) = (y[i] * step, -y[j] * step);
    alpha[i] += di;
    alpha[j] += dj;
    for t in 0..n {
      grad[t] += y[t] * (y[i] * kernel[t][i] * di + y[j] * kernel[t][j] * dj);
    }
  }
  // Bias from the free support vectors (or the midpoint of the bounds).
  let (mut sum, mut cnt) = (0.0, 0.0);
  let (mut ub, mut lb) = (f64::INFINITY, f64::NEG_INFINITY);
  for t in 0..n {
    let v = -y[t] * grad[t];
    if alpha[t] > 1e-9 && alpha[t] < c - 1e-9 {
      sum += v;
      cnt += 1.0;
    } else if (y[t] > 0.0) == (alpha[t] <= 1e-9) {
      ub = ub.min(v);
    } else {
      lb = lb.max(v);
    }
  }
  let b = if cnt > 0.0 {
    sum / cnt
  } else if ub.is_finite() && lb.is_finite() {
    f64::midpoint(ub, lb)
  } else {
    0.0
  };
  (alpha.iter().zip(y).map(|(a, yi)| a * yi).collect(), b)
}

/// Fit `P(y = 1 | f) = 1 / (1 + exp(A f + B))` by Newton's method on the
/// decision values (Platt scaling with Lin et al.'s target smoothing).
fn platt(f: &[f64], y: &[f64]) -> (f64, f64) {
  let pos = y.iter().filter(|&&v| v > 0.0).count() as f64;
  let neg = y.len() as f64 - pos;
  let hi = (pos + 1.0) / (pos + 2.0);
  let lo = 1.0 / (neg + 2.0);
  let t: Vec<f64> = y.iter().map(|&v| if v > 0.0 { hi } else { lo }).collect();
  let (mut a, mut b) = (0.0, ((neg + 1.0) / (pos + 1.0)).ln());
  for _ in 0..100 {
    let (mut h11, mut h22, mut h21, mut g1, mut g2) =
      (1e-12, 1e-12, 0.0, 0.0, 0.0);
    for (fi, ti) in f.iter().zip(&t) {
      let p = 1.0 / (1.0 + (a * fi + b).exp());
      let w = p * (1.0 - p);
      h11 += fi * fi * w;
      h22 += w;
      h21 += fi * w;
      g1 += fi * (ti - p);
      g2 += ti - p;
    }
    let det = h11 * h22 - h21 * h21;
    if det.abs() < 1e-300 {
      break;
    }
    let da = -(h22 * g1 - h21 * g2) / det;
    let db = -(-h21 * g1 + h11 * g2) / det;
    a += da;
    b += db;
    if da.abs() < 1e-10 && db.abs() < 1e-10 {
      break;
    }
  }
  (a, b)
}

fn train_svm(data: &Dataset, m: usize) -> Model {
  let scaler = Scaler::fit(&data.x);
  let x: Vec<Vec<f64>> = data.x.iter().map(|r| scaler.apply(r)).collect();
  let gamma = 1.0 / x[0].len() as f64;
  let kernel: Vec<Vec<f64>> = x
    .iter()
    .map(|a| x.iter().map(|b| rbf(a, b, gamma)).collect())
    .collect();
  // A binary problem needs one machine; more classes use one-vs-rest.
  let targets: Vec<usize> = if m == 2 { vec![1] } else { (0..m).collect() };
  let machines = targets
    .into_iter()
    .map(|c| {
      let y: Vec<f64> = data
        .y
        .iter()
        .map(|&v| if v == c { 1.0 } else { -1.0 })
        .collect();
      let (coef, b) = smo(&kernel, &y, 1.0);
      let f: Vec<f64> = kernel
        .iter()
        .map(|row| row.iter().zip(&coef).map(|(k, a)| k * a).sum::<f64>() + b)
        .collect();
      let (pa, pb) = platt(&f, &y);
      (coef, b, pa, pb)
    })
    .collect();
  Model::Svm {
    scaler,
    x,
    gamma,
    machines,
  }
}

fn train_forest(data: &Dataset, m: usize, trees: usize) -> Model {
  let n = data.x.len();
  let d = data.x[0].len();
  let mut rng = Rng(0x5EED_0002);
  if trees == 1 {
    let idx: Vec<usize> = (0..n).collect();
    return Model::Forest(vec![grow_tree(data, &idx, m, d, &mut rng, 0)]);
  }
  let n_try = ((d as f64).sqrt().round() as usize).max(1);
  let forest = (0..trees)
    .map(|_| {
      let idx: Vec<usize> = (0..n).map(|_| rng.below(n)).collect();
      grow_tree(data, &idx, m, n_try, &mut rng, 0)
    })
    .collect();
  Model::Forest(forest)
}

fn train(method: &str, data: &Dataset, m: usize) -> Model {
  match method {
    "NaiveBayes" => train_naive_bayes(data, m),
    "NearestNeighbors" => train_neighbors(data),
    "NeuralNetwork" => train_network(data, m),
    "SupportVectorMachine" => train_svm(data, m),
    "DecisionTree" => train_forest(data, m, 1),
    "RandomForest" => train_forest(data, m, 50),
    _ => train_logistic(data, m),
  }
}

impl Model {
  /// Class probabilities for one (raw, unscaled) feature vector.
  fn probabilities(&self, raw: &[f64], m: usize) -> Vec<f64> {
    match self {
      Model::Logistic { scaler, w } => {
        let x = scaler.apply(raw);
        let d = x.len();
        let z: Vec<f64> = w
          .iter()
          .map(|wc| wc[d] + wc.iter().zip(&x).map(|(a, b)| a * b).sum::<f64>())
          .collect();
        softmax(&z)
      }
      Model::NaiveBayes { prior, mean, var } => {
        let z: Vec<f64> = (0..m)
          .map(|c| {
            prior[c].ln()
              + raw
                .iter()
                .enumerate()
                .map(|(j, v)| {
                  let s2 = var[c][j];
                  -0.5
                    * ((v - mean[c][j]).powi(2) / s2
                      + (2.0 * std::f64::consts::PI * s2).ln())
                })
                .sum::<f64>()
          })
          .collect();
        softmax(&z)
      }
      Model::Neighbors { scaler, x, y, k } => {
        let q = scaler.apply(raw);
        let mut order: Vec<(f64, usize)> =
          x.iter().zip(y).map(|(p, &c)| (sq_dist(p, &q), c)).collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut counts = vec![0.0; m];
        for &(_, c) in order.iter().take(*k) {
          counts[c] += 1.0;
        }
        smoothed(&counts, 0.1)
      }
      Model::Network {
        scaler,
        w1,
        b1,
        w2,
        b2,
      } => {
        let x = scaler.apply(raw);
        let hidden: Vec<f64> = w1
          .iter()
          .zip(b1)
          .map(|(w, b)| {
            (b + w.iter().zip(&x).map(|(a, v)| a * v).sum::<f64>()).tanh()
          })
          .collect();
        let z: Vec<f64> = w2
          .iter()
          .zip(b2)
          .map(|(w, b)| {
            b + w.iter().zip(&hidden).map(|(a, v)| a * v).sum::<f64>()
          })
          .collect();
        softmax(&z)
      }
      Model::Svm {
        scaler,
        x,
        gamma,
        machines,
      } => {
        let q = scaler.apply(raw);
        let k: Vec<f64> = x.iter().map(|p| rbf(p, &q, *gamma)).collect();
        let scores: Vec<f64> = machines
          .iter()
          .map(|(coef, b, pa, pb)| {
            let f = k.iter().zip(coef).map(|(a, c)| a * c).sum::<f64>() + b;
            1.0 / (1.0 + (pa * f + pb).exp())
          })
          .collect();
        if m == 2 {
          vec![1.0 - scores[0], scores[0]]
        } else {
          let s: f64 = scores.iter().sum::<f64>().max(1e-300);
          scores.into_iter().map(|v| v / s).collect()
        }
      }
      Model::Forest(trees) => {
        let mut p = vec![0.0; m];
        for t in trees {
          for (acc, v) in p.iter_mut().zip(t.probabilities(raw)) {
            *acc += v / trees.len() as f64;
          }
        }
        p
      }
    }
  }
}

// ─── Model cache ───────────────────────────────────────────────────────────

thread_local! {
  static MODELS: RefCell<HashMap<i128, std::rc::Rc<Model>>> =
    RefCell::new(HashMap::new());
  static NEXT_ID: RefCell<i128> = const { RefCell::new(1) };
}

fn cache_model(model: Model) -> i128 {
  let id = NEXT_ID.with(|n| {
    let mut n = n.borrow_mut();
    let id = *n;
    *n += 1;
    id
  });
  MODELS.with(|c| {
    let mut c = c.borrow_mut();
    if c.len() > 256 {
      c.clear();
    }
    c.insert(id, std::rc::Rc::new(model));
  });
  id
}

// ─── The ClassifierFunction object ─────────────────────────────────────────

/// A decoded `ClassifierFunction[<|…|>]`.
struct Classifier {
  method: String,
  classes: Vec<Expr>,
  dim: usize,
  model: std::rc::Rc<Model>,
}

fn assoc_get<'a>(entries: &'a [(Expr, Expr)], key: &str) -> Option<&'a Expr> {
  entries
    .iter()
    .find(|(k, _)| matches!(k, Expr::String(s) if s == key))
    .map(|(_, v)| v)
}

fn decode(obj_args: &[Expr]) -> Option<Classifier> {
  let [Expr::Association(entries)] = obj_args else {
    return None;
  };
  let Expr::String(method) = assoc_get(entries, "Method")? else {
    return None;
  };
  let Expr::List(classes) = assoc_get(entries, "Classes")? else {
    return None;
  };
  let classes: Vec<Expr> = classes.iter().cloned().collect();
  let Expr::List(examples) = assoc_get(entries, "Examples")? else {
    return None;
  };
  let Expr::List(labels) = assoc_get(entries, "Labels")? else {
    return None;
  };
  let x: Vec<Vec<f64>> =
    examples.iter().map(features).collect::<Option<_>>()?;
  let dim = x.first()?.len();
  let cached = match assoc_get(entries, "ModelID") {
    Some(Expr::Integer(id)) => MODELS.with(|c| c.borrow().get(id).cloned()),
    _ => None,
  };
  let model = if let Some(m) = cached {
    m
  } else {
    let y: Vec<usize> = labels
      .iter()
      .map(|c| class_index(&classes, c))
      .collect::<Option<_>>()?;
    let model = train(method, &Dataset { x, y }, classes.len());
    let id = cache_model(model);
    MODELS.with(|c| c.borrow().get(&id).cloned())?
  };
  Some(Classifier {
    method: method.clone(),
    classes,
    dim,
    model,
  })
}

impl Classifier {
  fn probabilities(&self, x: &[f64]) -> Vec<f64> {
    self.model.probabilities(x, self.classes.len())
  }

  fn decide(&self, x: &[f64]) -> Expr {
    let p = self.probabilities(x);
    let best = (0..p.len()).fold(0, |b, i| if p[i] > p[b] { i } else { b });
    self.classes[best].clone()
  }

  /// One example's features when `e` is a single example of the right
  /// length.
  fn example(&self, e: &Expr) -> Option<Vec<f64>> {
    features(e).filter(|v| v.len() == self.dim)
  }
}

/// `Classify[training, opts…]`.
pub fn classify_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // The training set comes first (it may itself be a rule
  // `{x…} -> {c…}`); options follow.
  let Some((spec, opts)) = args.split_first() else {
    return Ok(unevaluated("Classify", args));
  };
  if !opts.iter().all(is_option) {
    return Ok(unevaluated("Classify", args));
  }
  let mut method = "Automatic".to_string();
  for opt in opts {
    if let Some((Expr::Identifier(k), v)) = rule_parts(opt)
      && k == "Method"
    {
      method = match v {
        Expr::String(s) => s.clone(),
        Expr::Identifier(s) if s == "Automatic" => s.clone(),
        Expr::List(items) => match items.first() {
          Some(Expr::String(s)) => s.clone(),
          _ => return Ok(unevaluated("Classify", args)),
        },
        _ => return Ok(unevaluated("Classify", args)),
      };
    }
  }
  if method == "Automatic" {
    method = "LogisticRegression".to_string();
  }
  if !METHODS.contains(&method.as_str()) {
    crate::emit_message(&format!(
      "Classify::bdmtd: Value of option Method -> \"{method}\" is not \
       Automatic or one of {}.",
      METHODS
        .iter()
        .map(|m| format!("\"{m}\""))
        .collect::<Vec<_>>()
        .join(", ")
    ));
    return Ok(unevaluated("Classify", args));
  }
  let Some(pairs) = labeled_pairs(spec) else {
    return Ok(unevaluated("Classify", args));
  };
  let Some(x) = pairs
    .iter()
    .map(|(e, _)| features(e))
    .collect::<Option<Vec<_>>>()
  else {
    return Ok(unevaluated("Classify", args));
  };
  let dim = x.first().map_or(0, Vec::len);
  if dim == 0 || x.iter().any(|r| r.len() != dim) {
    return Ok(unevaluated("Classify", args));
  }
  let mut classes: Vec<Expr> = Vec::new();
  for (_, c) in &pairs {
    if class_index(&classes, c).is_none() {
      classes.push(c.clone());
    }
  }
  classes.sort_by(|a, b| compare_exprs(a, b).cmp(&0).reverse());
  let y: Vec<usize> = pairs
    .iter()
    .map(|(_, c)| class_index(&classes, c).unwrap_or(0))
    .collect();
  let model = if classes.len() == 1 {
    None
  } else {
    Some(train(&method, &Dataset { x, y }, classes.len()))
  };
  let id = model.map_or_else(
    || cache_model(Model::Forest(vec![Tree::Leaf(vec![1.0])])),
    cache_model,
  );
  let examples: Vec<Expr> = pairs.iter().map(|(e, _)| e.clone()).collect();
  let labels: Vec<Expr> = pairs.into_iter().map(|(_, c)| c).collect();
  Ok(call1(
    "ClassifierFunction",
    Expr::Association(vec![
      (Expr::String("Method".into()), Expr::String(method)),
      (Expr::String("Classes".into()), Expr::List(classes.into())),
      (Expr::String("Examples".into()), Expr::List(examples.into())),
      (Expr::String("Labels".into()), Expr::List(labels.into())),
      (Expr::String("ModelID".into()), Expr::Integer(id)),
    ]),
  ))
}

fn probability_assoc(c: &Classifier, p: &[f64]) -> Expr {
  Expr::Association(
    c.classes
      .iter()
      .cloned()
      .zip(p.iter().map(|v| Expr::Real(*v)))
      .collect(),
  )
}

/// Answer property `prop` for one example.
fn example_property(c: &Classifier, x: &[f64], prop: &Expr) -> Option<Expr> {
  match prop {
    Expr::String(s) if s == "Decision" => Some(c.decide(x)),
    Expr::String(s) if s == "Probabilities" => {
      Some(probability_assoc(c, &c.probabilities(x)))
    }
    Expr::String(s) if s == "TopProbabilities" => {
      Some(top_probabilities(c, x, usize::MAX))
    }
    Expr::List(items) => match items.as_ref() {
      [Expr::String(s), class] if s == "Probability" => {
        let i = class_index(&c.classes, class)?;
        Some(Expr::Real(c.probabilities(x)[i]))
      }
      [Expr::String(s), n] if s == "TopProbabilities" => {
        let Expr::Integer(n) = n else { return None };
        Some(top_probabilities(c, x, usize::try_from(*n).ok()?))
      }
      _ => None,
    },
    _ => None,
  }
}

/// The classes with a non-negligible probability, most probable first.
fn top_probabilities(c: &Classifier, x: &[f64], n: usize) -> Expr {
  let p = c.probabilities(x);
  let mut order: Vec<usize> = (0..p.len()).collect();
  order.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
  let rules: Vec<Expr> = order
    .into_iter()
    .filter(|&i| p[i] >= 0.01)
    .take(n)
    .map(|i| Expr::Rule {
      pattern: Box::new(c.classes[i].clone()),
      replacement: Box::new(Expr::Real(p[i])),
    })
    .collect();
  Expr::List(rules.into())
}

/// `ClassifierFunction[…][input]` and `ClassifierFunction[…][input, prop]`.
pub fn apply_classifier(obj_args: &[Expr], args: &[Expr]) -> Option<Expr> {
  let c = decode(obj_args)?;
  match args {
    [Expr::String(s)] => match s.as_str() {
      "Classes" => Some(Expr::List(c.classes.clone().into())),
      "Method" => Some(Expr::String(c.method.clone())),
      "Properties" => Some(Expr::List(
        ["Classes", "Method", "Properties"]
          .iter()
          .map(|s| Expr::String((*s).into()))
          .collect::<Vec<_>>()
          .into(),
      )),
      _ => None,
    },
    [input] | [input, _] => {
      let prop = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| Expr::String("Decision".into()));
      if let Some(x) = c.example(input) {
        return example_property(&c, &x, &prop);
      }
      // A list of examples maps over them.
      let Expr::List(items) = input else {
        return None;
      };
      let out = items
        .iter()
        .map(|e| c.example(e).and_then(|x| example_property(&c, &x, &prop)))
        .collect::<Option<Vec<_>>>()?;
      Some(Expr::List(out.into()))
    }
    _ => None,
  }
}

// ─── ClassifierMeasurements ────────────────────────────────────────────────

const MEASURES: &[&str] = &[
  "Accuracy",
  "AccuracyBaseline",
  "ConfusionMatrix",
  "Error",
  "F1Score",
  "MeanCrossEntropy",
  "Precision",
  "Properties",
  "Recall",
];

/// `ClassifierMeasurements[c, test]` and `ClassifierMeasurements[c, test,
/// prop]`.
pub fn classifier_measurements_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let keep = || Ok(unevaluated("ClassifierMeasurements", args));
  let (obj, test, prop) = match args {
    [o, t] => (o, t, None),
    [o, t, p] => (o, t, Some(p)),
    _ => return keep(),
  };
  let Expr::FunctionCall { name, args: cargs } = obj else {
    return keep();
  };
  if name != "ClassifierFunction" || decode(cargs).is_none() {
    return keep();
  }
  let Some(pairs) = labeled_pairs(test) else {
    return keep();
  };
  let object = call1(
    "ClassifierMeasurementsObject",
    Expr::Association(vec![
      (Expr::String("Classifier".into()), obj.clone()),
      (
        Expr::String("Examples".into()),
        Expr::List(
          pairs
            .iter()
            .map(|(e, _)| e.clone())
            .collect::<Vec<_>>()
            .into(),
        ),
      ),
      (
        Expr::String("Labels".into()),
        Expr::List(
          pairs.into_iter().map(|(_, c)| c).collect::<Vec<_>>().into(),
        ),
      ),
    ]),
  );
  match prop {
    None => Ok(object),
    Some(p) => {
      let Expr::FunctionCall { args: oargs, .. } = &object else {
        unreachable!()
      };
      Ok(
        apply_measurements(oargs, std::slice::from_ref(p))
          .unwrap_or_else(|| unevaluated("ClassifierMeasurements", args)),
      )
    }
  }
}

/// `ClassifierMeasurementsObject[…][prop]`.
pub fn apply_measurements(obj_args: &[Expr], args: &[Expr]) -> Option<Expr> {
  let [Expr::Association(entries)] = obj_args else {
    return None;
  };
  let Expr::FunctionCall { args: cargs, .. } =
    assoc_get(entries, "Classifier")?
  else {
    return None;
  };
  let c = decode(cargs)?;
  let Expr::List(examples) = assoc_get(entries, "Examples")? else {
    return None;
  };
  let Expr::List(labels) = assoc_get(entries, "Labels")? else {
    return None;
  };
  let m = c.classes.len();
  let mut truth = Vec::new();
  let mut probs = Vec::new();
  for (e, l) in examples.iter().zip(labels.iter()) {
    truth.push(class_index(&c.classes, l)?);
    probs.push(c.probabilities(&c.example(e)?));
  }
  let pred: Vec<usize> = probs
    .iter()
    .map(|p| (0..m).fold(0, |b, i| if p[i] > p[b] { i } else { b }))
    .collect();
  let n = truth.len() as f64;
  if n == 0.0 {
    return None;
  }
  let mut confusion = vec![vec![0usize; m]; m];
  for (&t, &p) in truth.iter().zip(&pred) {
    confusion[t][p] += 1;
  }
  let correct: usize = (0..m).map(|i| confusion[i][i]).sum();
  let accuracy = correct as f64 / n;
  let per_class = |f: &dyn Fn(usize) -> f64| {
    Expr::Association(
      c.classes
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, k)| (k, Expr::Real(f(i))))
        .collect(),
    )
  };
  let precision = |i: usize| {
    let col: usize = (0..m).map(|t| confusion[t][i]).sum();
    if col == 0 {
      0.0
    } else {
      confusion[i][i] as f64 / col as f64
    }
  };
  let recall = |i: usize| {
    let row: usize = confusion[i].iter().sum();
    if row == 0 {
      0.0
    } else {
      confusion[i][i] as f64 / row as f64
    }
  };
  let [Expr::String(prop)] = args else {
    return None;
  };
  Some(match prop.as_str() {
    "Accuracy" => Expr::Real(accuracy),
    "Error" => Expr::Real(1.0 - accuracy),
    "AccuracyBaseline" => {
      let most = (0..m)
        .map(|i| truth.iter().filter(|&&t| t == i).count())
        .max()
        .unwrap_or(0);
      Expr::Real(most as f64 / n)
    }
    "ConfusionMatrix" => Expr::List(
      confusion
        .iter()
        .map(|row| {
          Expr::List(
            row
              .iter()
              .map(|&v| Expr::Integer(v as i128))
              .collect::<Vec<_>>()
              .into(),
          )
        })
        .collect::<Vec<_>>()
        .into(),
    ),
    "Precision" => per_class(&precision),
    "Recall" => per_class(&recall),
    "F1Score" => per_class(&|i| {
      let (p, r) = (precision(i), recall(i));
      if p + r == 0.0 {
        0.0
      } else {
        2.0 * p * r / (p + r)
      }
    }),
    "MeanCrossEntropy" => Expr::Real(
      -truth
        .iter()
        .zip(&probs)
        .map(|(&t, p)| p[t].max(1e-300).ln())
        .sum::<f64>()
        / n,
    ),
    "Properties" => Expr::List(
      MEASURES
        .iter()
        .map(|s| Expr::String((*s).into()))
        .collect::<Vec<_>>()
        .into(),
    ),
    _ => return None,
  })
}
