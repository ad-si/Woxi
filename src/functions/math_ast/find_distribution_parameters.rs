//! `FindDistributionParameters[data, dist]`: maximum-likelihood estimates
//! of the symbolic parameters of `dist` — in closed form where there is
//! one, numerically otherwise.

use crate::InterpreterError;
use crate::functions::math_ast::try_eval_to_f64;
use crate::syntax::Expr;

fn call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

fn rule(p: &Expr, v: f64) -> Expr {
  Expr::Rule {
    pattern: Box::new(p.clone()),
    replacement: Box::new(Expr::Real(v)),
  }
}

fn unevaluated(args: &[Expr]) -> Expr {
  call("FindDistributionParameters", args.to_vec())
}

/// Whether `e` is a parameter to estimate rather than a fixed value.
fn is_parameter(e: &Expr) -> bool {
  try_eval_to_f64(e).is_none()
}

pub fn find_distribution_parameters_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Ok(unevaluated(args));
  }
  let Expr::List(items) = &args[0] else {
    return Ok(unevaluated(args));
  };
  let Some(data) = items
    .iter()
    .map(try_eval_to_f64)
    .collect::<Option<Vec<f64>>>()
  else {
    return Ok(unevaluated(args));
  };
  let Expr::FunctionCall {
    name: dist_name,
    args: dist_args,
  } = &args[1]
  else {
    return Ok(unevaluated(args));
  };
  if data.is_empty() {
    return Ok(unevaluated(args));
  }
  // Optional starting values `{{p, p0}, …}`; options follow.
  let mut starts: Vec<(Expr, f64)> = Vec::new();
  for extra in &args[2..] {
    match extra {
      Expr::List(specs) => {
        for spec in specs {
          let Expr::List(pair) = spec else {
            return Ok(unevaluated(args));
          };
          let [p, v] = pair.as_slice() else {
            return Ok(unevaluated(args));
          };
          let Some(v) = try_eval_to_f64(v) else {
            return Ok(unevaluated(args));
          };
          starts.push((p.clone(), v));
        }
      }
      Expr::Rule { pattern, .. }
        if matches!(pattern.as_ref(), Expr::Identifier(n)
          if n == "ParameterEstimator" || n == "AccuracyGoal" || n == "PrecisionGoal" || n == "WorkingPrecision") =>
        {}
      _ => return Ok(unevaluated(args)),
    }
  }
  let params: Vec<usize> = (0..dist_args.len())
    .filter(|&i| is_parameter(&dist_args[i]))
    .collect();
  if params.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }
  if let Some(rules) = closed_form(dist_name, dist_args, &params, &data) {
    return Ok(Expr::List(rules.into()));
  }
  // Trial points outside the parameters' domain are expected on the way;
  // their messages are not the user's.
  crate::push_quiet();
  let result = numeric_mle(args, dist_name, dist_args, &params, &data, &starts);
  crate::pop_quiet();
  result
}

/// The maximum-likelihood estimates with a formula.
fn closed_form(
  name: &str,
  dist_args: &[Expr],
  params: &[usize],
  data: &[f64],
) -> Option<Vec<Expr>> {
  let n = data.len() as f64;
  let mean = data.iter().sum::<f64>() / n;
  let all = params.len() == dist_args.len();
  match (name, dist_args.len()) {
    ("NormalDistribution", 2) if all => {
      let sd =
        (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
      Some(vec![rule(&dist_args[0], mean), rule(&dist_args[1], sd)])
    }
    ("NormalDistribution", 2) if params == [0] => {
      Some(vec![rule(&dist_args[0], mean)])
    }
    ("LogNormalDistribution", 2) if all && data.iter().all(|&x| x > 0.0) => {
      let logs: Vec<f64> = data.iter().map(|x| x.ln()).collect();
      let mu = logs.iter().sum::<f64>() / n;
      let sigma =
        (logs.iter().map(|l| (l - mu).powi(2)).sum::<f64>() / n).sqrt();
      Some(vec![rule(&dist_args[0], mu), rule(&dist_args[1], sigma)])
    }
    ("ExponentialDistribution", 1) if all => {
      Some(vec![rule(&dist_args[0], 1.0 / mean)])
    }
    ("PoissonDistribution", 1) if all => Some(vec![rule(&dist_args[0], mean)]),
    // The likelihood of UniformDistribution[{a, b}] grows as the interval
    // shrinks onto the data.
    ("UniformDistribution", 1) => {
      let Expr::List(bounds) = &dist_args[0] else {
        return None;
      };
      let [a, b] = bounds.as_slice() else {
        return None;
      };
      if !is_parameter(a) || !is_parameter(b) {
        return None;
      }
      let lo = data.iter().copied().fold(f64::INFINITY, f64::min);
      let hi = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
      Some(vec![rule(a, lo), rule(b, hi)])
    }
    ("BinomialDistribution", 2) => binomial_mle(dist_args, params, data, mean),
    // The mean -θ/((1 - θ) Log[1 - θ]) increases with θ on (0, 1).
    ("LogSeriesDistribution", 1) if all && mean > 1.0 => {
      let mean_at = |t: f64| -t / ((1.0 - t) * (1.0 - t).ln());
      let (mut lo, mut hi) = (1e-15, 1.0 - 1e-15);
      for _ in 0..200 {
        let mid = f64::midpoint(lo, hi);
        if mean_at(mid) > mean {
          hi = mid;
        } else {
          lo = mid;
        }
      }
      Some(vec![rule(&dist_args[0], f64::midpoint(lo, hi))])
    }
    ("LaplaceDistribution", 2) if all => {
      let mut sorted = data.to_vec();
      sorted.sort_by(f64::total_cmp);
      let m = sorted.len();
      let median = if m % 2 == 1 {
        sorted[m / 2]
      } else {
        f64::midpoint(sorted[m / 2 - 1], sorted[m / 2])
      };
      let scale = data.iter().map(|x| (x - median).abs()).sum::<f64>() / n;
      Some(vec![
        rule(&dist_args[0], median),
        rule(&dist_args[1], scale),
      ])
    }
    _ => None,
  }
}

/// Maximum-likelihood estimates for `BinomialDistribution[n, p]` from
/// non-negative integer counts: p = mean / n for a known n, and otherwise
/// the integer n ≥ max(data) of highest (profile) likelihood.
fn binomial_mle(
  dist_args: &[Expr],
  params: &[usize],
  data: &[f64],
  mean: f64,
) -> Option<Vec<Expr>> {
  if data.iter().any(|&x| x < 0.0 || x.fract() != 0.0) {
    return None;
  }
  let max = data.iter().copied().fold(0.0, f64::max);
  let log_likelihood = |n: f64, p: f64| -> f64 {
    data
      .iter()
      .map(|&x| {
        ln_gamma(n + 1.0) - ln_gamma(x + 1.0) - ln_gamma(n - x + 1.0)
          + x * p.ln()
          + (n - x) * (1.0 - p).ln()
      })
      .sum()
  };
  // The best integer n in [max, cap], with p given as a function of n.
  let best_n = |p_of: &dyn Fn(f64) -> f64| -> Option<f64> {
    let cap = (100.0 * max).max(1000.0);
    let mut best: Option<(f64, f64)> = None;
    let mut n = max.max(1.0);
    while n <= cap {
      let p = p_of(n);
      if p > 0.0 && p < 1.0 {
        let ll = log_likelihood(n, p);
        if best.is_none_or(|(_, b)| ll > b) {
          best = Some((n, ll));
        }
      }
      n += 1.0;
    }
    // A maximum at the cap means the likelihood keeps growing: no estimate.
    best.filter(|(n, _)| *n < cap).map(|(n, _)| n)
  };
  let int_rule = |p: &Expr, v: f64| Expr::Rule {
    pattern: Box::new(p.clone()),
    replacement: Box::new(Expr::Integer(v as i128)),
  };
  match params {
    [1] => {
      let n = try_eval_to_f64(&dist_args[0])?;
      (n >= max && n > 0.0).then(|| vec![rule(&dist_args[1], mean / n)])
    }
    [0] => {
      let p = try_eval_to_f64(&dist_args[1])?;
      let n = best_n(&|_| p)?;
      Some(vec![int_rule(&dist_args[0], n)])
    }
    [0, 1] => {
      let n = best_n(&|n| mean / n)?;
      Some(vec![
        int_rule(&dist_args[0], n),
        rule(&dist_args[1], mean / n),
      ])
    }
    _ => None,
  }
}

/// The logarithm of the gamma function for positive arguments (Lanczos).
fn ln_gamma(x: f64) -> f64 {
  const G: f64 = 7.0;
  const C: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
  ];
  if x < 0.5 {
    let pi = std::f64::consts::PI;
    return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
  }
  let x = x - 1.0;
  let mut a = C[0];
  let t = x + G + 0.5;
  for (i, c) in C.iter().enumerate().skip(1) {
    a += c / (x + i as f64);
  }
  0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Maximize the log-likelihood numerically: a Nelder–Mead search from the
/// given starting values (1 for the others), then Newton steps on the
/// score to settle the last digits.
fn numeric_mle(
  args: &[Expr],
  dist_name: &str,
  dist_args: &[Expr],
  params: &[usize],
  data: &[f64],
  starts: &[(Expr, f64)],
) -> Result<Expr, InterpreterError> {
  // The density as an expression in a placeholder point, evaluated once.
  let point = Expr::Identifier("Woxi`FindDistributionParametersPoint".into());
  let names: Vec<String> = params
    .iter()
    .map(|&i| format!("Woxi`FindDistributionParameter{i}"))
    .collect();
  let mut placeholder_args = dist_args.to_vec();
  for (k, &i) in params.iter().enumerate() {
    placeholder_args[i] = Expr::Identifier(names[k].clone());
  }
  let density = crate::evaluator::evaluate_expr_to_expr(&call(
    "PDF",
    vec![call(dist_name, placeholder_args), point.clone()],
  ))?;
  if matches!(&density, Expr::FunctionCall { name, .. } if name == "PDF") {
    return Ok(unevaluated(args));
  }
  let log_likelihood = |values: &[f64]| -> f64 {
    let mut at_params = density.clone();
    for (name, v) in names.iter().zip(values) {
      at_params =
        crate::syntax::substitute_variable(&at_params, name, &Expr::Real(*v));
    }
    let Ok(at_params) = crate::evaluator::evaluate_expr_to_expr(&at_params)
    else {
      return f64::NEG_INFINITY;
    };
    let mut total = 0.0;
    for &x in data {
      let at = crate::syntax::substitute_variable(
        &at_params,
        "Woxi`FindDistributionParametersPoint",
        &Expr::Real(x),
      );
      let value = crate::evaluator::evaluate_expr_to_expr(&at)
        .ok()
        .and_then(|e| try_eval_to_f64(&e));
      match value {
        Some(p) if p > 0.0 && p.is_finite() => total += p.ln(),
        _ => return f64::NEG_INFINITY,
      }
    }
    total
  };
  let x0: Vec<f64> = params
    .iter()
    .map(|&i| {
      starts
        .iter()
        .find(|(p, _)| {
          crate::syntax::expr_to_string(p)
            == crate::syntax::expr_to_string(&dist_args[i])
        })
        .map_or(1.0, |(_, v)| *v)
    })
    .collect();
  let objective = |v: &[f64]| {
    let ll = log_likelihood(v);
    if ll.is_finite() { -ll } else { f64::INFINITY }
  };
  if !objective(&x0).is_finite() {
    return Ok(unevaluated(args));
  }
  let mut best = crate::functions::polynomial_ast::solve::nelder_mead_minimize(
    &objective,
    &|_: &mut [f64]| {},
    &x0,
    4000,
  );
  // Newton steps on the score — the exact derivatives of the summed
  // log-density, with the Hessian from differences of the score — settle
  // the digits the simplex search leaves.
  let positive_branch = match &density {
    Expr::FunctionCall { name, args: pw }
      if name == "Piecewise"
        && matches!(pw.first(), Some(Expr::List(cases)) if cases.len() == 1) =>
    {
      match &pw[0] {
        Expr::List(cases) => match &cases[0] {
          Expr::List(case) if case.len() == 2 => case[0].clone(),
          _ => density.clone(),
        },
        _ => density.clone(),
      }
    }
    _ => density.clone(),
  };
  let log_density = call("Log", vec![positive_branch]);
  let scores: Option<Vec<Expr>> = names
    .iter()
    .map(|name| {
      crate::evaluator::evaluate_expr_to_expr(&call(
        "D",
        vec![log_density.clone(), Expr::Identifier(name.clone())],
      ))
      .ok()
    })
    .collect();
  if let Some(scores) = scores {
    let score = |values: &[f64]| -> Option<Vec<f64>> {
      scores
        .iter()
        .map(|d| {
          let mut at_params = d.clone();
          for (name, v) in names.iter().zip(values) {
            at_params = crate::syntax::substitute_variable(
              &at_params,
              name,
              &Expr::Real(*v),
            );
          }
          let at_params =
            crate::evaluator::evaluate_expr_to_expr(&at_params).ok()?;
          let mut total = 0.0;
          for &x in data {
            let at = crate::syntax::substitute_variable(
              &at_params,
              "Woxi`FindDistributionParametersPoint",
              &Expr::Real(x),
            );
            total += crate::evaluator::evaluate_expr_to_expr(&at)
              .ok()
              .and_then(|e| try_eval_to_f64(&e))
              .filter(|v| v.is_finite())?;
          }
          Some(total)
        })
        .collect()
    };
    let m = best.len();
    for _ in 0..30 {
      let Some(g) = score(&best) else {
        break;
      };
      let mut hess = vec![vec![0.0; m]; m];
      let mut ok = true;
      for j in 0..m {
        let h = 1e-6 * best[j].abs().max(1e-6);
        let mut up = best.clone();
        let mut down = best.clone();
        up[j] += h;
        down[j] -= h;
        match (score(&up), score(&down)) {
          (Some(gu), Some(gd)) => {
            for i in 0..m {
              hess[i][j] = (gu[i] - gd[i]) / (2.0 * h);
            }
          }
          _ => ok = false,
        }
      }
      if !ok {
        break;
      }
      let Some(step) = solve_small(&hess, &g) else {
        break;
      };
      let candidate: Vec<f64> =
        best.iter().zip(&step).map(|(b, s)| b - s).collect();
      if !objective(&candidate).is_finite() {
        break;
      }
      let size = step
        .iter()
        .zip(&best)
        .map(|(s, b)| (s / b.abs().max(1e-12)).abs())
        .fold(0.0, f64::max);
      best = candidate;
      if size < 1e-15 {
        break;
      }
    }
  }
  Ok(Expr::List(
    params
      .iter()
      .zip(&best)
      .map(|(&i, &v)| rule(&dist_args[i], v))
      .collect::<Vec<_>>()
      .into(),
  ))
}

/// Solve the small dense system `a x = b` by Gaussian elimination.
fn solve_small(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
  let n = b.len();
  let mut m: Vec<Vec<f64>> = a
    .iter()
    .zip(b)
    .map(|(row, &bi)| {
      let mut r = row.clone();
      r.push(bi);
      r
    })
    .collect();
  for c in 0..n {
    let pivot =
      (c..n).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs()))?;
    if m[pivot][c].abs() < 1e-300 {
      return None;
    }
    m.swap(c, pivot);
    for r in 0..n {
      if r != c {
        let f = m[r][c] / m[c][c];
        for k in c..=n {
          m[r][k] -= f * m[c][k];
        }
      }
    }
  }
  Some((0..n).map(|i| m[i][n] / m[i][i]).collect())
}

/// `EstimatedDistribution[data, dist, …]`: `dist` with the estimates of
/// `FindDistributionParameters[data, dist, …]` in place of its parameters.
pub fn estimated_distribution_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Ok(call("EstimatedDistribution", args.to_vec()));
  }
  // Data in one unit fit the distribution of their magnitudes, which is
  // then a distribution of quantities in that unit.
  if let Expr::List(items) = &args[0]
    && let Some((magnitudes, unit)) = common_unit(items)
    && !matches!(&args[1], Expr::FunctionCall { name, .. }
      if name == "QuantityDistribution")
  {
    let mut rest = args.to_vec();
    rest[0] = Expr::List(magnitudes.into());
    let fitted = estimated_distribution_ast(&rest)?;
    if matches!(&fitted, Expr::FunctionCall { name, .. }
      if name == "EstimatedDistribution")
    {
      return Ok(call("EstimatedDistribution", args.to_vec()));
    }
    return Ok(call("QuantityDistribution", vec![fitted, unit]));
  }
  let rules = find_distribution_parameters_ast(args)?;
  if !matches!(&rules, Expr::List(_)) {
    return Ok(call("EstimatedDistribution", args.to_vec()));
  }
  crate::evaluator::evaluate_expr_to_expr(&call(
    "ReplaceAll",
    vec![args[1].clone(), rules],
  ))
}

/// The magnitudes of `items` when all are quantities in the same unit,
/// and that unit.
fn common_unit(items: &[Expr]) -> Option<(Vec<Expr>, Expr)> {
  let mut unit: Option<&Expr> = None;
  let mut magnitudes = Vec::with_capacity(items.len());
  for item in items {
    let Expr::FunctionCall { name, args } = item else {
      return None;
    };
    if name != "Quantity" || args.len() != 2 {
      return None;
    }
    match unit {
      None => unit = Some(&args[1]),
      Some(u)
        if crate::syntax::expr_to_string(u)
          == crate::syntax::expr_to_string(&args[1]) => {}
      Some(_) => return None,
    }
    magnitudes.push(args[0].clone());
  }
  Some((magnitudes, unit?.clone()))
}
