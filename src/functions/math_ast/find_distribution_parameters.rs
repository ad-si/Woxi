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
  let rules = find_distribution_parameters_ast(args)?;
  if !matches!(&rules, Expr::List(_)) {
    return Ok(call("EstimatedDistribution", args.to_vec()));
  }
  crate::evaluator::evaluate_expr_to_expr(&call(
    "ReplaceAll",
    vec![args[1].clone(), rules],
  ))
}
