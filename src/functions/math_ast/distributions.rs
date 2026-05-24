#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::syntax::{BinaryOperator, ComparisonOp, Expr};

/// Helper to build a binary operation expression
fn binop(op: BinaryOperator, left: Expr, right: Expr) -> Expr {
  Expr::BinaryOp {
    op,
    left: Box::new(left),
    right: Box::new(right),
  }
}

fn times(a: Expr, b: Expr) -> Expr {
  binop(BinaryOperator::Times, a, b)
}

fn divide(a: Expr, b: Expr) -> Expr {
  binop(BinaryOperator::Divide, a, b)
}

fn power(base: Expr, exp: Expr) -> Expr {
  binop(BinaryOperator::Power, base, exp)
}

fn minus(a: Expr, b: Expr) -> Expr {
  binop(BinaryOperator::Minus, a, b)
}

fn sqrt(a: Expr) -> Expr {
  make_sqrt(a)
}

fn factorial(a: Expr) -> Expr {
  Expr::FunctionCall {
    name: "Factorial".to_string(),
    args: vec![a].into(),
  }
}

fn e() -> Expr {
  Expr::Constant("E".to_string())
}

fn pi() -> Expr {
  Expr::Constant("Pi".to_string())
}

fn int(n: i128) -> Expr {
  Expr::Integer(n)
}

fn piecewise(pairs: Vec<(Expr, Expr)>, default: Expr) -> Expr {
  let cases = Expr::List(
    pairs
      .into_iter()
      .map(|(val, cond)| Expr::List(vec![val, cond].into()))
      .collect(),
  );
  Expr::FunctionCall {
    name: "Piecewise".to_string(),
    args: vec![cases, default].into(),
  }
}

fn comparison(left: Expr, op: ComparisonOp, right: Expr) -> Expr {
  Expr::Comparison {
    operands: vec![left, right],
    operators: vec![op],
  }
}

fn comparison3(
  a: Expr,
  op1: ComparisonOp,
  b: Expr,
  op2: ComparisonOp,
  c: Expr,
) -> Expr {
  Expr::Comparison {
    operands: vec![a, b, c],
    operators: vec![op1, op2],
  }
}

fn eval(expr: Expr) -> Result<Expr, InterpreterError> {
  crate::evaluator::evaluate_expr_to_expr(&expr)
}

/// PDF[dist, x] - Probability density function
/// PDF[dist] - Pure function form (returns unevaluated for now)
pub fn pdf_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "PDF expects 1 or 2 arguments".into(),
    ));
  }

  let dist = &args[0];

  // Extract distribution name and parameters
  let (dist_name, dargs) = match dist {
    Expr::FunctionCall { name, args: dargs } => {
      (name.as_str(), dargs.as_slice())
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PDF".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  if args.len() == 1 {
    // PDF[dist] - return unevaluated for now
    return Ok(Expr::FunctionCall {
      name: "PDF".to_string(),
      args: args.to_vec().into(),
    });
  }

  let x = args[1].clone();

  match dist_name {
    "ProbabilityDistribution" => pdf_probability_distribution(dargs, x),
    "NormalDistribution" => pdf_normal(dargs, x),
    "UniformDistribution" => pdf_uniform(dargs, x),
    "ExponentialDistribution" => pdf_exponential(dargs, x),
    "PoissonDistribution" => pdf_poisson(dargs, x),
    "BernoulliDistribution" => pdf_bernoulli(dargs, x),
    "InverseGammaDistribution" => pdf_inverse_gamma(dargs, x),
    "GammaDistribution" => pdf_gamma(dargs, x),
    "BetaDistribution" => pdf_beta(dargs, x),
    "StudentTDistribution" => pdf_student_t(dargs, x),
    "LogNormalDistribution" => pdf_lognormal(dargs, x),
    "ChiSquareDistribution" => pdf_chi_square(dargs, x),
    "ParetoDistribution" => pdf_pareto(dargs, x),
    "WeibullDistribution" => pdf_weibull(dargs, x),
    "GeometricDistribution" => pdf_geometric(dargs, x),
    "CauchyDistribution" => pdf_cauchy(dargs, x),
    "DiscreteUniformDistribution" => pdf_discrete_uniform(dargs, x),
    "LaplaceDistribution" => pdf_laplace(dargs, x),
    "RayleighDistribution" => pdf_rayleigh(dargs, x),
    "MultinomialDistribution" => pdf_multinomial(dargs, x),
    "NegativeBinomialDistribution" => pdf_negative_binomial(dargs, x),
    "HalfNormalDistribution" => pdf_half_normal(dargs, x),
    "ChiDistribution" => pdf_chi(dargs, x),
    "LogisticDistribution" => pdf_logistic(dargs, x),
    "InverseChiSquareDistribution" => pdf_inverse_chi_square(dargs, x),
    "FrechetDistribution" => pdf_frechet(dargs, x),
    "ExtremeValueDistribution" => pdf_extreme_value(dargs, x),
    "GompertzMakehamDistribution" => pdf_gompertz_makeham(dargs, x),
    "InverseGaussianDistribution" => pdf_inverse_gaussian(dargs, x),
    "StableDistribution" => pdf_stable(dargs, x),
    "ArcSinDistribution" => pdf_arcsin(dargs, x),
    "PascalDistribution" => pdf_pascal(dargs, x),
    "DagumDistribution" => pdf_dagum(dargs, x),
    "HyperbolicDistribution" => pdf_hyperbolic(dargs, x),
    "NoncentralFRatioDistribution" => pdf_noncentral_f(dargs, x),
    "JohnsonDistribution" => pdf_johnson(dargs, x),
    _ => Ok(Expr::FunctionCall {
      name: "PDF".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// `SurvivalFunction[dist, x] = 1 - CDF[dist, x]`.
pub fn survival_function_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "SurvivalFunction expects 1 or 2 arguments".into(),
    ));
  }
  if args.len() == 1 {
    // SurvivalFunction[dist] — symbolic pure form, leave unevaluated.
    return Ok(Expr::FunctionCall {
      name: "SurvivalFunction".to_string(),
      args: args.to_vec().into(),
    });
  }
  let cdf = cdf_ast(args)?;
  eval(minus(int(1), cdf))
}

/// Closed-form Quantile[dist, q] for distributions whose inverse CDF is
/// elementary. Returns `Some(expr)` when the head is recognised. Callers
/// should try this before the numerical fallback.
pub fn quantile_distribution_closed_form(
  dist_name: &str,
  dargs: &[Expr],
  q: &Expr,
) -> Option<Expr> {
  match dist_name {
    // Quantile[ExponentialDistribution[lambda], q] = -Log[1 - q] / lambda
    "ExponentialDistribution" if dargs.len() == 1 => {
      let lambda = dargs[0].clone();
      let one_minus_q = minus(int(1), q.clone());
      let log_term = Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![one_minus_q].into(),
      };
      let neg_log = times(int(-1), log_term);
      let expr = divide(neg_log, lambda);
      eval(expr).ok()
    }
    _ => None,
  }
}

/// Numerical inverse-CDF for distributions whose CDF Woxi can compute
/// symbolically or numerically. Returns `Some(quantile)` if successful,
/// `None` if the head isn't recognised (caller falls back), or `Err` for
/// hard failures.
pub fn quantile_distribution_numeric(
  dist_name: &str,
  dargs: &[Expr],
  q: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  // Only handle ProbabilityDistribution numerically for now — the standard
  // distributions have their own closed-form quantile elsewhere (or stay
  // symbolic). Extending to LogNormal/Pareto/etc. is a future task.
  if dist_name != "ProbabilityDistribution" {
    return Ok(None);
  }
  if dargs.len() != 2 {
    return Ok(None);
  }
  let q_val = match q {
    Expr::Integer(n) => *n as f64,
    Expr::Real(f) => *f,
    _ => return Ok(None),
  };
  if !(0.0..=1.0).contains(&q_val) {
    return Err(InterpreterError::EvaluationError(format!(
      "Quantile: probability {q_val} not in [0, 1]"
    )));
  }

  let Expr::List(items) = &dargs[1] else {
    return Ok(None);
  };
  if items.len() != 3 {
    return Ok(None);
  }
  let Expr::Identifier(_var) = &items[0] else {
    return Ok(None);
  };
  let lo = try_eval_to_f64(&items[1]).unwrap_or(0.0);
  let hi_expr = &items[2];
  // Bracket the upper bound; for Infinity, start with lo + 1 and expand.
  let mut hi_known = try_eval_to_f64(hi_expr);
  let hi_is_infinite = matches!(hi_expr, Expr::Identifier(n) if n == "Infinity")
    || hi_known.map(|h| h.is_infinite()).unwrap_or(false);
  if hi_is_infinite {
    hi_known = None;
  }
  let dist_expr = Expr::FunctionCall {
    name: "ProbabilityDistribution".to_string(),
    args: dargs.to_vec().into(),
  };
  // Helper: numeric CDF evaluation at `x`.
  let cdf_at = |x: f64| -> Option<f64> {
    let val = cdf_ast(&[dist_expr.clone(), Expr::Real(x)]).ok()?;
    try_eval_to_f64(&val)
  };

  // Establish an upper bracket. For infinite support, double until CDF ≥ q.
  let mut hi: f64 = match hi_known {
    Some(h) => h,
    None => {
      let mut h = (lo + 1.0).max(1.0);
      for _ in 0..60 {
        match cdf_at(h) {
          Some(c) if c >= q_val => break,
          _ => h *= 2.0,
        }
      }
      h
    }
  };
  let mut lo_b = lo;
  // Bisection: ~50 iterations gives ≈1e-15 precision for any sane support.
  for _ in 0..80 {
    let mid = 0.5 * (lo_b + hi);
    let Some(c) = cdf_at(mid) else {
      return Ok(None);
    };
    if c < q_val {
      lo_b = mid;
    } else {
      hi = mid;
    }
    if (hi - lo_b).abs() < 1e-14 {
      break;
    }
  }
  Ok(Some(Expr::Real(0.5 * (lo_b + hi))))
}

/// PDF[ProbabilityDistribution[pdf_expr, {var, lo, hi}], t] substitutes `t`
/// for the dummy variable in the pdf expression, returning a piecewise that
/// is zero outside `[lo, hi]`.
fn pdf_probability_distribution(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  use crate::functions::plot::substitute_var;
  if dargs.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "ProbabilityDistribution: missing pdf".into(),
    ));
  }
  let pdf = &dargs[0];
  // Univariate: single iterator. Multivariate PDF[…, {a, b}] isn't supported here.
  if dargs.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "PDF".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "ProbabilityDistribution".to_string(),
          args: dargs.to_vec().into(),
        },
        x,
      ]
      .into(),
    });
  }
  let Expr::List(items) = &dargs[1] else {
    return Ok(Expr::FunctionCall {
      name: "PDF".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "ProbabilityDistribution".to_string(),
          args: dargs.to_vec().into(),
        },
        x,
      ]
      .into(),
    });
  };
  if items.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "PDF".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "ProbabilityDistribution".to_string(),
          args: dargs.to_vec().into(),
        },
        x,
      ]
      .into(),
    });
  }
  let Expr::Identifier(var) = &items[0] else {
    return Ok(Expr::FunctionCall {
      name: "PDF".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "ProbabilityDistribution".to_string(),
          args: dargs.to_vec().into(),
        },
        x,
      ]
      .into(),
    });
  };
  let lo = items[1].clone();
  let hi = items[2].clone();
  let density = substitute_var(pdf, var, &x);
  let cond =
    comparison3(lo, ComparisonOp::LessEqual, x, ComparisonOp::LessEqual, hi);
  eval(piecewise(vec![(density, cond)], int(0)))
}

/// CDF[ProbabilityDistribution[pdf, {var, lo, hi}], t] = Integrate[pdf, {var, lo, t}].
/// Returns 0 for t < lo and 1 for t > hi (piecewise).
fn cdf_probability_distribution(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "CDF".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "ProbabilityDistribution".to_string(),
          args: dargs.to_vec().into(),
        },
        x,
      ]
      .into(),
    });
  }
  let pdf = &dargs[0];
  let Expr::List(items) = &dargs[1] else {
    return Ok(Expr::FunctionCall {
      name: "CDF".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "ProbabilityDistribution".to_string(),
          args: dargs.to_vec().into(),
        },
        x,
      ]
      .into(),
    });
  };
  if items.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "CDF".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "ProbabilityDistribution".to_string(),
          args: dargs.to_vec().into(),
        },
        x,
      ]
      .into(),
    });
  }
  let Expr::Identifier(var) = &items[0] else {
    return Ok(Expr::FunctionCall {
      name: "CDF".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "ProbabilityDistribution".to_string(),
          args: dargs.to_vec().into(),
        },
        x,
      ]
      .into(),
    });
  };
  let lo = items[1].clone();
  let integral = Expr::FunctionCall {
    name: "Integrate".to_string(),
    args: vec![
      pdf.clone(),
      Expr::List(vec![Expr::Identifier(var.clone()), lo, x].into()),
    ]
    .into(),
  };
  eval(integral)
}

/// PDF[NormalDistribution[mu, sigma], x] = 1/(E^((-mu + x)^2/(2*sigma^2))*Sqrt[2*Pi]*sigma)
fn pdf_normal(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let (mu, sigma) = match dargs.len() {
    0 | 2 => {
      let mu = if dargs.is_empty() {
        int(0)
      } else {
        dargs[0].clone()
      };
      let sigma = if dargs.is_empty() {
        int(1)
      } else {
        dargs[1].clone()
      };
      (mu, sigma)
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "NormalDistribution expects 0 or 2 arguments".into(),
      ));
    }
  };

  // 1/(E^((x - mu)^2/(2*sigma^2)) * Sqrt[2*Pi] * sigma)
  let diff = minus(x, mu);
  let diff_sq = power(diff, int(2));
  let two_sigma_sq = times(int(2), power(sigma.clone(), int(2)));
  let exponent = divide(diff_sq, two_sigma_sq);
  let exp_part = power(e(), exponent);
  let sqrt_part = sqrt(times(int(2), pi()));
  let denominator = times(times(exp_part, sqrt_part), sigma);
  let result = divide(int(1), denominator);
  eval(result)
}

/// PDF[UniformDistribution[{a, b}], x] = Piecewise[{{1/(b-a), a <= x <= b}}, 0]
fn pdf_uniform(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let (a, b) = match dargs.len() {
    0 => (int(0), int(1)),
    1 => {
      if let Expr::List(bounds) = &dargs[0] {
        if bounds.len() == 2 {
          (bounds[0].clone(), bounds[1].clone())
        } else {
          return Err(InterpreterError::EvaluationError(
            "UniformDistribution: expected {min, max}".into(),
          ));
        }
      } else {
        return Err(InterpreterError::EvaluationError(
          "UniformDistribution: expected a list {min, max}".into(),
        ));
      }
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "UniformDistribution expects 0 or 1 argument".into(),
      ));
    }
  };

  let density = eval(divide(int(1), minus(b.clone(), a.clone())))?;
  let cond =
    comparison3(a, ComparisonOp::LessEqual, x, ComparisonOp::LessEqual, b);

  eval(piecewise(vec![(density, cond)], int(0)))
}

/// PDF[ExponentialDistribution[lambda], x] = Piecewise[{{lambda*E^(-lambda*x), x >= 0}}, 0]
fn pdf_exponential(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ExponentialDistribution expects 1 argument".into(),
    ));
  }
  let lambda = dargs[0].clone();

  let density =
    eval(divide(lambda.clone(), power(e(), times(lambda, x.clone()))))?;
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));

  eval(piecewise(vec![(density, cond)], int(0)))
}

/// PDF[PoissonDistribution[mu], k] = Piecewise[{{mu^k/(E^mu * k!), k >= 0}}, 0]
fn pdf_poisson(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PoissonDistribution expects 1 argument".into(),
    ));
  }
  let mu = dargs[0].clone();

  let numerator = power(mu.clone(), x.clone());
  let denominator = times(power(e(), mu), factorial(x.clone()));
  let density = eval(divide(numerator, denominator))?;
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));

  eval(piecewise(vec![(density, cond)], int(0)))
}

fn plus(a: Expr, b: Expr) -> Expr {
  binop(BinaryOperator::Plus, a, b)
}

/// PDF[BernoulliDistribution[p], k] = Piecewise[{{1-p, k==0}, {p, k==1}}, 0]
fn pdf_bernoulli(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "BernoulliDistribution expects 1 argument".into(),
    ));
  }
  let p = dargs[0].clone();

  let one_minus_p = eval(minus(int(1), p.clone()))?;
  let cond0 = comparison(x.clone(), ComparisonOp::Equal, int(0));
  let cond1 = comparison(x, ComparisonOp::Equal, int(1));

  eval(piecewise(vec![(one_minus_p, cond0), (p, cond1)], int(0)))
}

/// PDF[CauchyDistribution[a, b], x] = 1/(Pi*b*(1+((x-a)/b)^2))
fn pdf_cauchy(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let (a, b) = match dargs.len() {
    0 => (int(0), int(1)),
    2 => (dargs[0].clone(), dargs[1].clone()),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "CauchyDistribution expects 0 or 2 arguments".into(),
      ));
    }
  };
  // 1 / (Pi * b * (1 + ((x - a) / b)^2))
  let diff = minus(x, a);
  let ratio = divide(diff, b.clone());
  let denom = times(times(pi(), b), plus(int(1), power(ratio, int(2))));
  eval(divide(int(1), denom))
}

/// PDF[GeometricDistribution[p], k] = Piecewise[{{(1-p)^k * p, k >= 0}}, 0]
fn pdf_geometric(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "GeometricDistribution expects 1 argument".into(),
    ));
  }
  let p = dargs[0].clone();
  let one_minus_p = minus(int(1), p.clone());
  let density = eval(times(power(one_minus_p, x.clone()), p))?;
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));
  eval(piecewise(vec![(density, cond)], int(0)))
}

// ─── CDF ──────────────────────────────────────────────────────────────

/// CDF[dist, x] - Cumulative distribution function
pub fn cdf_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "CDF expects 1 or 2 arguments".into(),
    ));
  }

  let dist = &args[0];

  let (dist_name, dargs) = match dist {
    Expr::FunctionCall { name, args: dargs } => {
      (name.as_str(), dargs.as_slice())
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "CDF".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  if args.len() == 1 {
    return Ok(Expr::FunctionCall {
      name: "CDF".to_string(),
      args: args.to_vec().into(),
    });
  }

  let x = args[1].clone();

  match dist_name {
    "ProbabilityDistribution" => cdf_probability_distribution(dargs, x),
    "NormalDistribution" => cdf_normal(dargs, x),
    "UniformDistribution" => cdf_uniform(dargs, x),
    "ExponentialDistribution" => cdf_exponential(dargs, x),
    "PoissonDistribution" => cdf_poisson(dargs, x),
    "BernoulliDistribution" => cdf_bernoulli(dargs, x),
    "InverseGammaDistribution" => cdf_inverse_gamma(dargs, x),
    "GammaDistribution" => cdf_gamma(dargs, x),
    "BetaDistribution" => cdf_beta(dargs, x),
    "LogNormalDistribution" => cdf_lognormal(dargs, x),
    "ChiSquareDistribution" => cdf_chi_square(dargs, x),
    "ParetoDistribution" => cdf_pareto(dargs, x),
    "WeibullDistribution" => cdf_weibull(dargs, x),
    "GeometricDistribution" => cdf_geometric(dargs, x),
    "CauchyDistribution" => cdf_cauchy(dargs, x),
    "DiscreteUniformDistribution" => cdf_discrete_uniform(dargs, x),
    "LaplaceDistribution" => cdf_laplace(dargs, x),
    "RayleighDistribution" => cdf_rayleigh(dargs, x),
    "HalfNormalDistribution" => cdf_half_normal(dargs, x),
    "ChiDistribution" => cdf_chi(dargs, x),
    "LogisticDistribution" => cdf_logistic(dargs, x),
    "InverseChiSquareDistribution" => cdf_inverse_chi_square(dargs, x),
    "FrechetDistribution" => cdf_frechet(dargs, x),
    "ExtremeValueDistribution" => cdf_extreme_value(dargs, x),
    "GompertzMakehamDistribution" => cdf_gompertz_makeham(dargs, x),
    "InverseGaussianDistribution" => cdf_inverse_gaussian(dargs, x),
    "StableDistribution" => cdf_stable(dargs, x),
    "StudentTDistribution" => cdf_student_t(dargs, x),
    "JohnsonDistribution" => cdf_johnson(dargs, x),
    _ => Ok(Expr::FunctionCall {
      name: "CDF".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// CDF[NormalDistribution[mu, sigma], x] = Erfc[(mu - x)/(Sqrt[2]*sigma)]/2
fn cdf_normal(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let (mu, sigma) = match dargs.len() {
    0 | 2 => {
      let mu = if dargs.is_empty() {
        int(0)
      } else {
        dargs[0].clone()
      };
      let sigma = if dargs.is_empty() {
        int(1)
      } else {
        dargs[1].clone()
      };
      (mu, sigma)
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "NormalDistribution expects 0 or 2 arguments".into(),
      ));
    }
  };

  // Erfc[(mu - x) / (Sqrt[2] * sigma)] / 2
  let erfc_arg = divide(minus(mu, x), times(sqrt(int(2)), sigma));
  let erfc_call = Expr::FunctionCall {
    name: "Erfc".to_string(),
    args: vec![erfc_arg].into(),
  };
  let result = divide(erfc_call, int(2));
  eval(result)
}

/// CDF[UniformDistribution[{a, b}], x] = Piecewise[{{(x-a)/(b-a), a<=x<=b}, {1, x>b}}, 0]
fn cdf_uniform(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let (a, b) = match dargs.len() {
    0 => (int(0), int(1)),
    1 => {
      if let Expr::List(bounds) = &dargs[0] {
        if bounds.len() == 2 {
          (bounds[0].clone(), bounds[1].clone())
        } else {
          return Err(InterpreterError::EvaluationError(
            "UniformDistribution: expected {min, max}".into(),
          ));
        }
      } else {
        return Err(InterpreterError::EvaluationError(
          "UniformDistribution: expected a list {min, max}".into(),
        ));
      }
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "UniformDistribution expects 0 or 1 argument".into(),
      ));
    }
  };

  let value = eval(divide(
    minus(x.clone(), a.clone()),
    minus(b.clone(), a.clone()),
  ))?;
  let cond_middle = comparison3(
    a,
    ComparisonOp::LessEqual,
    x.clone(),
    ComparisonOp::LessEqual,
    b.clone(),
  );
  let cond_above = comparison(x, ComparisonOp::Greater, b);

  eval(piecewise(
    vec![(value, cond_middle), (int(1), cond_above)],
    int(0),
  ))
}

/// CDF[ExponentialDistribution[lambda], x] = Piecewise[{{1 - E^(-lambda*x), x >= 0}}, 0]
fn cdf_exponential(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ExponentialDistribution expects 1 argument".into(),
    ));
  }
  let lambda = dargs[0].clone();

  let value = eval(minus(
    int(1),
    power(
      e(),
      times(
        Expr::UnaryOp {
          op: crate::syntax::UnaryOperator::Minus,
          operand: Box::new(lambda),
        },
        x.clone(),
      ),
    ),
  ))?;
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));

  eval(piecewise(vec![(value, cond)], int(0)))
}

/// CDF[PoissonDistribution[mu], k] = Piecewise[{{GammaRegularized[Floor[k]+1, mu], k >= 0}}, 0]
/// For integer k, this equals sum_{i=0}^{k} mu^i * E^(-mu) / i!
fn cdf_poisson(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PoissonDistribution expects 1 argument".into(),
    ));
  }
  let mu = dargs[0].clone();

  // GammaRegularized[Floor[k] + 1, mu]
  let floor_k_plus_1 = plus(
    Expr::FunctionCall {
      name: "Floor".to_string(),
      args: vec![x.clone()].into(),
    },
    int(1),
  );
  let value = Expr::FunctionCall {
    name: "GammaRegularized".to_string(),
    args: vec![floor_k_plus_1, mu].into(),
  };
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));

  let result = piecewise(vec![(value, cond)], int(0));
  eval(result)
}

/// CDF[BernoulliDistribution[p], k] = Piecewise[{{0, k<0}, {1-p, 0<=k<1}}, 1]
fn cdf_bernoulli(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "BernoulliDistribution expects 1 argument".into(),
    ));
  }
  let p = dargs[0].clone();

  let one_minus_p = eval(minus(int(1), p))?;
  let cond_neg = comparison(x.clone(), ComparisonOp::Less, int(0));
  let cond_middle = comparison3(
    int(0),
    ComparisonOp::LessEqual,
    x,
    ComparisonOp::Less,
    int(1),
  );

  eval(piecewise(
    vec![(int(0), cond_neg), (one_minus_p, cond_middle)],
    int(1),
  ))
}

/// CDF[GeometricDistribution[p], k] = Piecewise[{{1 - (1-p)^(Floor[k]+1), k >= 0}}, 0]
fn cdf_geometric(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "GeometricDistribution expects 1 argument".into(),
    ));
  }
  let p = dargs[0].clone();
  let one_minus_p = minus(int(1), p);
  let floor_k_plus_1 = plus(
    Expr::FunctionCall {
      name: "Floor".to_string(),
      args: vec![x.clone()].into(),
    },
    int(1),
  );
  let value = minus(int(1), power(one_minus_p, floor_k_plus_1));
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// CDF[CauchyDistribution[a, b], x] = 1/2 + ArcTan[(x-a)/b]/Pi
fn cdf_cauchy(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let (a, b) = match dargs.len() {
    0 => (int(0), int(1)),
    2 => (dargs[0].clone(), dargs[1].clone()),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "CauchyDistribution expects 0 or 2 arguments".into(),
      ));
    }
  };
  // 1/2 + ArcTan[(x - a) / b] / Pi
  let arctan = Expr::FunctionCall {
    name: "ArcTan".to_string(),
    args: vec![divide(minus(x, a), b)].into(),
  };
  eval(plus(
    Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![int(1), int(2)].into(),
    },
    divide(arctan, pi()),
  ))
}

/// PDF[GammaDistribution[alpha, beta], x] = Piecewise[{{x^(alpha-1) E^(-x/beta) / (beta^alpha Gamma[alpha]), x > 0}}, 0]
fn pdf_gamma(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "GammaDistribution expects 2 arguments".into(),
    ));
  }
  let alpha = dargs[0].clone();
  let beta = dargs[1].clone();

  // x^(alpha-1)
  let x_part = power(x.clone(), minus(alpha.clone(), int(1)));
  // E^(-x/beta)
  let exp_part = power(
    e(),
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(divide(x.clone(), beta.clone())),
    },
  );
  // beta^alpha * Gamma[alpha]
  let denom = times(
    power(beta, alpha.clone()),
    Expr::FunctionCall {
      name: "Gamma".to_string(),
      args: vec![alpha].into(),
    },
  );
  let value = eval(divide(times(x_part, exp_part), denom))?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// CDF[GammaDistribution[alpha, beta], x] = Piecewise[{{GammaRegularized[alpha, 0, x/beta], x > 0}}, 0]
fn cdf_gamma(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "GammaDistribution expects 2 arguments".into(),
    ));
  }
  let alpha = dargs[0].clone();
  let beta = dargs[1].clone();

  let value = Expr::FunctionCall {
    name: "GammaRegularized".to_string(),
    args: vec![alpha, int(0), divide(x.clone(), beta)].into(),
  };
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

// PDF[InverseGammaDistribution[a, b], x] = Piecewise[{{(b/x)^a/(E^(b/x)*x*Gamma[a]), x > 0}}, 0]
fn pdf_inverse_gamma(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseGammaDistribution expects 2 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();

  // (b/x)^a
  let bx_a = power(divide(b.clone(), x.clone()), a.clone());
  // E^(b/x)
  let exp_part = power(e(), divide(b, x.clone()));
  // x * Gamma[a]
  let denom = times(
    x.clone(),
    Expr::FunctionCall {
      name: "Gamma".to_string(),
      args: vec![a].into(),
    },
  );
  let value = eval(divide(bx_a, times(exp_part, denom)))?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

// CDF[InverseGammaDistribution[a, b], x] = Piecewise[{{GammaRegularized[a, b/x], x > 0}}, 0]
fn cdf_inverse_gamma(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseGammaDistribution expects 2 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();

  let value = Expr::FunctionCall {
    name: "GammaRegularized".to_string(),
    args: vec![a, divide(b, x.clone())].into(),
  };
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

// PDF[LogisticDistribution[m, s], x] = E^((m - x)/s)/((1 + E^((m - x)/s))^2*s)
fn pdf_logistic(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LogisticDistribution expects 2 arguments".into(),
    ));
  }
  let m = dargs[0].clone();
  let s = dargs[1].clone();

  // E^((m - x)/s)
  let exp_val = power(e(), divide(minus(m, x), s.clone()));
  // (1 + E^(...))^2
  let denom_sq = power(
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left: Box::new(int(1)),
      right: Box::new(exp_val.clone()),
    },
    int(2),
  );
  let value = divide(exp_val, times(denom_sq, s));
  eval(value)
}

// CDF[LogisticDistribution[m, s], x] = 1/(1 + E^((m - x)/s))
fn cdf_logistic(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LogisticDistribution expects 2 arguments".into(),
    ));
  }
  let m = dargs[0].clone();
  let s = dargs[1].clone();

  let exp_val = power(e(), divide(minus(m, x), s));
  let denom = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Plus,
    left: Box::new(int(1)),
    right: Box::new(exp_val),
  };
  eval(power(denom, int(-1)))
}

// PDF[InverseChiSquareDistribution[n], x] = Piecewise[{{(x^(-1))^(1+n/2)/(2^(n/2)*E^(1/(2*x))*Gamma[n/2]), x > 0}}, 0]
fn pdf_inverse_chi_square(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "InverseChiSquareDistribution expects 1 argument".into(),
    ));
  }
  let n = dargs[0].clone();

  // (x^(-1))^(1 + n/2) / (2^(n/2) * E^(1/(2*x)) * Gamma[n/2])
  // Use BinaryOp Power for (x^(-1))^(1+n/2) which the evaluator flattens.
  // Then wrap in divide which puts it in the numerator.
  let x_part = power(
    power(x.clone(), int(-1)),
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left: Box::new(int(1)),
      right: Box::new(divide(n.clone(), int(2))),
    },
  );
  // 2^(n/2)
  let two_part = power(int(2), divide(n.clone(), int(2)));
  // E^(1/(2*x))
  let exp_part = power(e(), divide(int(1), times(int(2), x.clone())));
  // Gamma[n/2]
  let gamma_part = Expr::FunctionCall {
    name: "Gamma".to_string(),
    args: vec![divide(n, int(2))].into(),
  };
  let value =
    eval(divide(x_part, times(two_part, times(exp_part, gamma_part))))?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

// CDF[InverseChiSquareDistribution[n], x] = Piecewise[{{GammaRegularized[n/2, 1/(2*x)], x > 0}}, 0]
fn cdf_inverse_chi_square(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "InverseChiSquareDistribution expects 1 argument".into(),
    ));
  }
  let n = dargs[0].clone();

  let value = Expr::FunctionCall {
    name: "GammaRegularized".to_string(),
    args: vec![divide(n, int(2)), divide(int(1), times(int(2), x.clone()))]
      .into(),
  };
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

// PDF[FrechetDistribution[a, b], x] = Piecewise[{{(a*(x/b)^(-1 - a))/(b*E^(x/b)^(-a)), x > 0}}, 0]
fn pdf_frechet(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "FrechetDistribution expects 2 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();

  let xb = divide(x.clone(), b.clone());
  // (x/b)^(-1 - a)
  let xb_part = power(
    xb.clone(),
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(int(1)),
        right: Box::new(a.clone()),
      }),
    },
  );
  // E^(x/b)^(-a)
  let exp_part = power(
    e(),
    power(
      xb,
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(a.clone()),
      },
    ),
  );
  let value = eval(divide(times(a, xb_part), times(b, exp_part)))?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

// CDF[FrechetDistribution[a, b], x] = Piecewise[{{E^(-(x/b)^(-a)), x > 0}}, 0]
fn cdf_frechet(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "FrechetDistribution expects 2 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();

  let value = power(
    e(),
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(power(
        divide(x.clone(), b),
        Expr::UnaryOp {
          op: crate::syntax::UnaryOperator::Minus,
          operand: Box::new(a),
        },
      )),
    },
  );
  let value = eval(value)?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

// PDF[ExtremeValueDistribution[a, b], x] = E^(-E^((a - x)/b) + (a - x)/b)/b
fn pdf_extreme_value(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ExtremeValueDistribution expects 2 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();

  let ab = divide(minus(a.clone(), x), b.clone());
  // E^(-E^((a-x)/b) + (a-x)/b)
  let exp_arg = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Plus,
    left: Box::new(Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(power(e(), ab.clone())),
    }),
    right: Box::new(ab),
  };
  let exp_arg_eval = eval(exp_arg)?;
  eval(divide(power(e(), exp_arg_eval), b))
}

// CDF[ExtremeValueDistribution[a, b], x] = E^(-E^((a - x)/b))
fn cdf_extreme_value(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ExtremeValueDistribution expects 2 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();

  let ab = divide(minus(a, x), b);
  eval(power(
    e(),
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(power(e(), ab)),
    },
  ))
}

// PDF[GompertzMakehamDistribution[l, x0], x] = Piecewise[{{E^(l*x + (1 - E^(l*x))*x0)*l*x0, x >= 0}}, 0]
fn pdf_gompertz_makeham(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "GompertzMakehamDistribution expects 2 arguments".into(),
    ));
  }
  let l = dargs[0].clone();
  let x0 = dargs[1].clone();

  // l*x
  let lx = times(l.clone(), x.clone());
  // E^(l*x)
  let e_lx = power(e(), lx.clone());
  // (1 - E^(l*x))*x0
  let inner = times(minus(int(1), e_lx), x0.clone());
  // l*x + (1 - E^(l*x))*x0
  let exp_arg = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Plus,
    left: Box::new(lx),
    right: Box::new(inner),
  };
  let value = eval(times(times(power(e(), exp_arg), l), x0))?;
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

// CDF[GompertzMakehamDistribution[l, x0], x] = Piecewise[{{1 - E^((1 - E^(l*x))*x0), x >= 0}}, 0]
fn cdf_gompertz_makeham(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "GompertzMakehamDistribution expects 2 arguments".into(),
    ));
  }
  let l = dargs[0].clone();
  let x0 = dargs[1].clone();

  let e_lx = power(e(), times(l, x.clone()));
  let inner = times(minus(int(1), e_lx), x0);
  let value = minus(int(1), power(e(), inner));
  let value = eval(value)?;
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

// PDF[InverseGaussianDistribution[m, l], x] = Piecewise[{{Sqrt[l/x^3]/(E^((l*(-m+x)^2)/(2*m^2*x))*Sqrt[2*Pi]), x > 0}}, 0]
fn pdf_inverse_gaussian(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseGaussianDistribution expects 2 arguments".into(),
    ));
  }
  let m = dargs[0].clone();
  let l = dargs[1].clone();

  // Sqrt[l/x^3]
  let numer = sqrt(divide(l.clone(), power(x.clone(), int(3))));
  // l*(-m+x)^2 / (2*m^2*x)
  let exp_arg = divide(
    times(l, power(minus(x.clone(), m.clone()), int(2))),
    times(int(2), times(power(m, int(2)), x.clone())),
  );
  // E^(exp_arg)
  let exp_part = power(e(), exp_arg);
  // Sqrt[2*Pi]
  let sqrt_2pi = sqrt(times(int(2), pi()));
  let value = eval(divide(numer, times(exp_part, sqrt_2pi)))?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

// CDF[InverseGaussianDistribution[m, l], x]
fn cdf_inverse_gaussian(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseGaussianDistribution expects 2 arguments".into(),
    ));
  }
  let m = dargs[0].clone();
  let l = dargs[1].clone();

  // Sqrt[l/x]
  let sqrt_lx = sqrt(divide(l.clone(), x.clone()));
  // Erfc[((m - x)*Sqrt[l/x])/(Sqrt[2]*m)]/2
  let erfc1_arg = divide(
    times(minus(m.clone(), x.clone()), sqrt_lx.clone()),
    times(sqrt(int(2)), m.clone()),
  );
  let erfc1 = divide(
    Expr::FunctionCall {
      name: "Erfc".to_string(),
      args: vec![erfc1_arg].into(),
    },
    int(2),
  );
  // E^((2*l)/m) * Erfc[(Sqrt[l/x]*(m + x))/(Sqrt[2]*m)]/2
  let exp_part = power(e(), divide(times(int(2), l), m.clone()));
  let erfc2_arg = divide(
    times(
      sqrt_lx,
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(m.clone()),
        right: Box::new(x.clone()),
      },
    ),
    times(sqrt(int(2)), m),
  );
  let erfc2 = divide(
    Expr::FunctionCall {
      name: "Erfc".to_string(),
      args: vec![erfc2_arg].into(),
    },
    int(2),
  );
  let value = eval(Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Plus,
    left: Box::new(erfc1),
    right: Box::new(times(exp_part, erfc2)),
  })?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

// ─── Probability ─────────────────────────────────────────────────────

/// Probability[event, x \[Distributed] dist]
/// Probability[event, x \[Distributed] d1 && y \[Distributed] d2 && ...]
/// event can be: x > a, x < a, x >= a, x <= a, x == k, a < x < b, etc.
pub fn probability_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Probability expects 2 arguments".into(),
    ));
  }

  let event = &args[0];
  let dist_spec = &args[1];

  // Conditional probability: P[A \[Conditioned] B, x ~ d] = P[A && B, x ~ d] / P[B, x ~ d].
  if let Expr::FunctionCall { name, args: cargs } = event
    && name == "Conditioned"
    && cargs.len() == 2
  {
    let joint = Expr::FunctionCall {
      name: "And".to_string(),
      args: vec![cargs[0].clone(), cargs[1].clone()].into(),
    };
    let p_joint = probability_ast(&[joint, dist_spec.clone()])?;
    let p_cond = probability_ast(&[cargs[1].clone(), dist_spec.clone()])?;
    return eval(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(p_joint),
      right: Box::new(p_cond),
    });
  }

  // Joint distribution form:
  //   x \[Distributed] d1 && y \[Distributed] d2 && ...
  // Handle it by enumerating the finite discrete support of every variable.
  if let Some(pairs) = collect_distributed_pairs(dist_spec)
    && pairs.len() >= 2
  {
    if let Some(result) = try_joint_probability_discrete(event, &pairs)? {
      return eval(Expr::FunctionCall {
        name: "Together".to_string(),
        args: vec![result].into(),
      });
    }
    return Ok(Expr::FunctionCall {
      name: "Probability".to_string(),
      args: args.to_vec().into(),
    });
  }

  // Parse Distributed[var, dist] from the second argument
  let (var_name, dist) = match dist_spec {
    Expr::FunctionCall { name, args: dargs }
      if name == "Distributed" && dargs.len() == 2 =>
    {
      if let Expr::Identifier(v) = &dargs[0] {
        (v.as_str(), &dargs[1])
      } else {
        return Ok(Expr::FunctionCall {
          name: "Probability".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Probability".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // P[event, x \[Distributed] ProbabilityDistribution[pdf, {x, lo, hi}]]
  // = Integrate[pdf * Boole[event], {x, lo, hi}], reduced to a sub-range
  // integral for simple `x > k` / `x < k` / `a < x < b` events.
  if let Expr::FunctionCall { name, args: dargs } = dist
    && name == "ProbabilityDistribution"
    && let Some(result) =
      try_probability_probability_distribution(event, var_name, dargs)?
  {
    return Ok(result);
  }

  // Determine if distribution is discrete
  let is_discrete = matches!(
    dist,
    Expr::FunctionCall { name, .. }
    if matches!(name.as_str(), "PoissonDistribution" | "BernoulliDistribution" | "BinomialDistribution" | "GeometricDistribution" | "NegativeBinomialDistribution" | "DiscreteUniformDistribution" | "PascalDistribution")
  );

  // Parse the event condition and compute probability
  let result = probability_from_event(event, var_name, dist, is_discrete)?;
  // Apply Together to normalize fractions (e.g. 1 - E^(-2) → (-1 + E^2)/E^2)
  eval(Expr::FunctionCall {
    name: "Together".to_string(),
    args: vec![result].into(),
  })
}

fn probability_from_event(
  event: &Expr,
  var: &str,
  dist: &Expr,
  is_discrete: bool,
) -> Result<Expr, InterpreterError> {
  // Handle And conditions: a < x && x < b  →  a < x < b
  if let Expr::FunctionCall { name, args } = event
    && name == "And"
    && args.len() == 2
  {
    // Try to extract compound inequality: lower < x && x < upper
    if let (Some((lo, lo_op)), Some((hi, hi_op))) = (
      extract_bound(&args[0], var, true),
      extract_bound(&args[1], var, false),
    ) {
      // P[lo < x < hi] = CDF[dist, hi] - CDF[dist, lo]
      let cdf_hi = cdf_ast(&[dist.clone(), hi])?;
      let cdf_lo = cdf_ast(&[dist.clone(), lo])?;
      let result = minus(cdf_hi, cdf_lo);
      // Adjust for inclusive/exclusive bounds if needed (for continuous, same)
      let _ = (lo_op, hi_op); // For continuous distributions, < vs <= doesn't matter
      return eval(result);
    }
    // Try reversed: x < upper && lower < x
    if let (Some((hi, hi_op)), Some((lo, lo_op))) = (
      extract_bound(&args[0], var, false),
      extract_bound(&args[1], var, true),
    ) {
      let cdf_hi = cdf_ast(&[dist.clone(), hi])?;
      let cdf_lo = cdf_ast(&[dist.clone(), lo])?;
      let result = minus(cdf_hi, cdf_lo);
      let _ = (lo_op, hi_op);
      return eval(result);
    }
  }

  // Handle Comparison: a < x, x > a, a <= x <= b, etc.
  if let Expr::Comparison {
    operands,
    operators,
  } = event
  {
    // Two-operand comparison: x > a, x < a, x >= a, x <= a, x == a
    if operands.len() == 2 && operators.len() == 1 {
      let op = &operators[0];
      let (left, right) = (&operands[0], &operands[1]);

      let is_left_var = matches!(left, Expr::Identifier(n) if n == var);
      let is_right_var = matches!(right, Expr::Identifier(n) if n == var);

      // x == k → PDF[dist, k] for discrete distributions
      if *op == ComparisonOp::Equal {
        if is_left_var {
          return pdf_ast(&[dist.clone(), right.clone()]);
        }
        if is_right_var {
          return pdf_ast(&[dist.clone(), left.clone()]);
        }
      }

      // x > a → 1 - CDF[dist, a]
      if is_left_var
        && (*op == ComparisonOp::Greater || *op == ComparisonOp::GreaterEqual)
      {
        let cdf_val = cdf_ast(&[dist.clone(), right.clone()])?;
        return eval(minus(int(1), cdf_val));
      }
      // x < a → CDF[dist, a]
      if is_left_var
        && (*op == ComparisonOp::Less || *op == ComparisonOp::LessEqual)
      {
        return cdf_ast(&[dist.clone(), right.clone()]);
      }
      // a > x → CDF[dist, a]  (same as x < a)
      if is_right_var
        && (*op == ComparisonOp::Greater || *op == ComparisonOp::GreaterEqual)
      {
        return cdf_ast(&[dist.clone(), left.clone()]);
      }
      // a < x → 1 - CDF[dist, a]
      if is_right_var
        && (*op == ComparisonOp::Less || *op == ComparisonOp::LessEqual)
      {
        let cdf_val = cdf_ast(&[dist.clone(), left.clone()])?;
        return eval(minus(int(1), cdf_val));
      }
    }

    // Three-operand comparison: a < x < b, a <= x <= b
    if operands.len() == 3 && operators.len() == 2 {
      let (lo, mid, hi) = (&operands[0], &operands[1], &operands[2]);
      if matches!(mid, Expr::Identifier(n) if n == var) {
        // lo < x < hi → CDF[dist, hi] - CDF[dist, lo]
        let both_less = matches!(
          (&operators[0], &operators[1]),
          (
            ComparisonOp::Less | ComparisonOp::LessEqual,
            ComparisonOp::Less | ComparisonOp::LessEqual
          )
        );
        if both_less {
          let cdf_hi = cdf_ast(&[dist.clone(), hi.clone()])?;
          let cdf_lo = cdf_ast(&[dist.clone(), lo.clone()])?;
          return eval(minus(cdf_hi, cdf_lo));
        }
      }
    }
  }

  let _ = is_discrete;

  // Unevaluated fallback
  Ok(Expr::FunctionCall {
    name: "Probability".to_string(),
    args: vec![
      event.clone(),
      Expr::FunctionCall {
        name: "Distributed".to_string(),
        args: vec![Expr::Identifier(var.to_string()), dist.clone()].into(),
      },
    ]
    .into(),
  })
}

/// Walk an `And[Distributed[x1, d1], Distributed[x2, d2], ...]` expression
/// (or a single `Distributed[x, d]`) and collect every `(var, dist)` pair.
/// Returns `None` if any branch isn't a well-formed `Distributed[...]`.
fn collect_distributed_pairs(expr: &Expr) -> Option<Vec<(String, Expr)>> {
  fn walk(expr: &Expr, out: &mut Vec<(String, Expr)>) -> bool {
    match expr {
      Expr::FunctionCall { name, args } if name == "And" => {
        args.iter().all(|a| walk(a, out))
      }
      Expr::FunctionCall { name, args }
        if name == "Distributed" && args.len() == 2 =>
      {
        if let Expr::Identifier(v) = &args[0] {
          out.push((v.clone(), args[1].clone()));
          true
        } else {
          false
        }
      }
      _ => false,
    }
  }
  let mut out = Vec::new();
  if walk(expr, &mut out) && !out.is_empty() {
    Some(out)
  } else {
    None
  }
}

/// Return the finite integer support of a discrete distribution as
/// `(Vec<Expr>, point_probability)` if and only if we can enumerate it
/// cheaply. Currently supports `DiscreteUniformDistribution[{a, b}]`.
fn discrete_finite_support(dist: &Expr) -> Option<(Vec<Expr>, Expr)> {
  let Expr::FunctionCall { name, args } = dist else {
    return None;
  };
  match name.as_str() {
    "DiscreteUniformDistribution" => {
      if args.len() != 1 {
        return None;
      }
      let Expr::List(range) = &args[0] else {
        return None;
      };
      if range.len() != 2 {
        return None;
      }
      let (Expr::Integer(lo), Expr::Integer(hi)) = (&range[0], &range[1])
      else {
        return None;
      };
      if hi < lo {
        return None;
      }
      let count = hi - lo + 1;
      let support: Vec<Expr> = (*lo..=*hi).map(Expr::Integer).collect();
      let prob = Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(count)].into(),
      };
      Some((support, prob))
    }
    _ => None,
  }
}

/// Attempt to compute a joint discrete probability by enumerating the
/// Cartesian product of each variable's finite support. Returns `Ok(None)`
/// if any distribution isn't a supported discrete finite distribution.
fn try_joint_probability_discrete(
  event: &Expr,
  pairs: &[(String, Expr)],
) -> Result<Option<Expr>, InterpreterError> {
  // Collect (support, point_prob) for every variable.
  let mut per_var: Vec<(String, Vec<Expr>, Expr)> =
    Vec::with_capacity(pairs.len());
  for (var, dist) in pairs {
    let Some((support, prob)) = discrete_finite_support(dist) else {
      return Ok(None);
    };
    per_var.push((var.clone(), support, prob));
  }

  // Accumulate probability mass for all points where the event evaluates
  // to True after substituting the sampled values for each variable.
  let mut total: Expr = Expr::Integer(0);
  // Iterate the Cartesian product via an index vector.
  let sizes: Vec<usize> = per_var.iter().map(|(_, s, _)| s.len()).collect();
  let mut idx = vec![0usize; sizes.len()];
  loop {
    // Build the substituted event.
    let mut substituted = event.clone();
    let mut point_prob: Expr = Expr::Integer(1);
    for (i, (var, support, pprob)) in per_var.iter().enumerate() {
      substituted = crate::functions::plot::substitute_var(
        &substituted,
        var,
        &support[idx[i]],
      );
      point_prob = eval(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![point_prob, pprob.clone()].into(),
      })?;
    }
    let evaluated = evaluate_expr_to_expr(&substituted)?;
    let is_true = matches!(&evaluated, Expr::Identifier(n) if n == "True");
    if is_true {
      total = eval(Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![total, point_prob].into(),
      })?;
    }

    // Advance the index vector (little-endian odometer).
    let mut i = 0;
    loop {
      if i == sizes.len() {
        return Ok(Some(total));
      }
      idx[i] += 1;
      if idx[i] < sizes[i] {
        break;
      }
      idx[i] = 0;
      i += 1;
    }
  }
}

/// Extract a bound from a comparison involving the variable.
/// If `lower` is true, look for patterns like "a < x" or "a <= x" (lower bound = a).
/// If `lower` is false, look for patterns like "x < b" or "x <= b" (upper bound = b).
/// Returns (bound_value, operator).
fn extract_bound(
  expr: &Expr,
  var: &str,
  lower: bool,
) -> Option<(Expr, ComparisonOp)> {
  if let Expr::Comparison {
    operands,
    operators,
  } = expr
    && operands.len() == 2
    && operators.len() == 1
  {
    let op = &operators[0];
    let (left, right) = (&operands[0], &operands[1]);
    let is_left_var = matches!(left, Expr::Identifier(n) if n == var);
    let is_right_var = matches!(right, Expr::Identifier(n) if n == var);

    if lower {
      // Looking for: a < x or a <= x (lower bound is a)
      if is_right_var
        && matches!(op, ComparisonOp::Less | ComparisonOp::LessEqual)
      {
        return Some((left.clone(), *op));
      }
      // x > a or x >= a (lower bound is a)
      if is_left_var
        && matches!(op, ComparisonOp::Greater | ComparisonOp::GreaterEqual)
      {
        return Some((right.clone(), *op));
      }
    } else {
      // Looking for: x < b or x <= b (upper bound is b)
      if is_left_var
        && matches!(op, ComparisonOp::Less | ComparisonOp::LessEqual)
      {
        return Some((right.clone(), *op));
      }
      // b > x or b >= x (upper bound is b)
      if is_right_var
        && matches!(op, ComparisonOp::Greater | ComparisonOp::GreaterEqual)
      {
        return Some((left.clone(), *op));
      }
    }
  }
  None
}

// ─── Expectation ─────────────────────────────────────────────────────

/// Expectation[f(x), x \[Distributed] dist]
/// Computes E[f(x)] for known distributions.
pub fn expectation_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Expectation expects 2 arguments".into(),
    ));
  }

  let expr = &args[0];
  let dist_spec = &args[1];

  // Parse Distributed[vars, dist] — vars may be a single Identifier or a List
  // (for multivariate distributions such as ProbabilityDistribution[..., {x,…}, {y,…}]).
  let (vars, dist) = match dist_spec {
    Expr::FunctionCall { name, args: dargs }
      if name == "Distributed" && dargs.len() == 2 =>
    {
      let vars = match &dargs[0] {
        Expr::Identifier(v) => vec![v.clone()],
        Expr::List(items) => {
          let mut names = Vec::with_capacity(items.len());
          for item in items.iter() {
            if let Expr::Identifier(n) = item {
              names.push(n.clone());
            } else {
              return Ok(Expr::FunctionCall {
                name: "Expectation".to_string(),
                args: args.to_vec().into(),
              });
            }
          }
          names
        }
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Expectation".to_string(),
            args: args.to_vec().into(),
          });
        }
      };
      (vars, &dargs[1])
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Expectation".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Get distribution name and parameters
  let (dist_name, dargs) = match dist {
    Expr::FunctionCall { name, args: da } => (name.as_str(), da.as_slice()),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Expectation".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // ProbabilityDistribution[pdf, {x, lo, hi}, …] — evaluate as
  // Integrate[expr*pdf, {x, lo, hi}, …] after renaming the distribution's
  // dummy variables to the names supplied in `Distributed[vars, …]`.
  if dist_name == "ProbabilityDistribution"
    && let Some(result) =
      try_expectation_probability_distribution(expr, &vars, dargs)?
  {
    return Ok(result);
  }

  // CensoredDistribution[{a, b}, base] truncates `base` to the interval
  // [a, b], piling mass that falls outside onto the boundary. So
  //   E[f(x), x ~ CensoredDistribution[{a, b}, base]]
  //     = f(a)*P[X < a] + ∫_a^b f(x)*pdf_base(x) dx + f(b)*P[X > b]
  // where the boundary probabilities come from base's CDF.
  if dist_name == "CensoredDistribution"
    && vars.len() == 1
    && let Some(result) = try_expectation_censored(expr, &vars[0], dargs)?
  {
    return Ok(result);
  }

  // For unsupported parameterizations (e.g. list-of-vars over a standard
  // distribution) just return unevaluated.
  if vars.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "Expectation".to_string(),
      args: args.to_vec().into(),
    });
  }
  let var_name = vars.into_iter().next().unwrap();

  // Compute E[f(x)] using known moment formulas
  // First try to get mean and variance for common distributions
  let (mean, variance) = distribution_mean_variance(dist_name, dargs)?;

  // Check if expr is just the variable (E[x] = mean)
  if matches!(expr, Expr::Identifier(n) if *n == var_name) {
    return eval(mean);
  }

  // Check if expr is x^2 (E[x^2] = Var + Mean^2)
  if is_power_of_var(expr, &var_name, 2) {
    let result = plus(variance.clone(), power(mean.clone(), int(2)));
    return eval(result);
  }

  // Check for linear expressions: a*x + b
  if let Some((a, b)) = extract_linear(expr, &var_name) {
    // E[a*x + b] = a*E[x] + b
    let result = plus(times(a, mean), b);
    return eval(result);
  }

  // For more complex expressions, use numerical integration
  expectation_numerical(expr, &var_name, dist_name, dargs)
}

/// Compute `Probability[event, x \[Distributed] ProbabilityDistribution[pdf, {x, lo, hi}]]`
/// by integrating the pdf over the sub-range carved out by `event`.
///
/// Supported event shapes (single variable, single iterator only):
///   - `x > k`, `x >= k`        → Integrate[pdf, {x, max(k,lo), hi}]
///   - `x < k`, `x <= k`        → Integrate[pdf, {x, lo, min(k,hi)}]
///   - `a < x < b`, `a ≤ x ≤ b` → Integrate[pdf, {x, max(a,lo), min(b,hi)}]
///   - `x \[Conditioned] y`     → P[x ∧ y] / P[y]
/// Returns `None` if the shape isn't recognised so the caller can fall back.
fn try_probability_probability_distribution(
  event: &Expr,
  var_name: &str,
  dargs: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  use crate::functions::plot::substitute_var;

  if dargs.len() != 2 {
    return Ok(None);
  }
  let pdf_raw = &dargs[0];
  let iter = &dargs[1];

  let Expr::List(iter_items) = iter else {
    return Ok(None);
  };
  if iter_items.len() != 3 {
    return Ok(None);
  }
  let Expr::Identifier(dummy) = &iter_items[0] else {
    return Ok(None);
  };
  let lo = &iter_items[1];
  let hi = &iter_items[2];

  // Rename the distribution's dummy var to the user-supplied var_name in the pdf.
  let pdf = if dummy == var_name {
    pdf_raw.clone()
  } else {
    substitute_var(pdf_raw, dummy, &Expr::Identifier(var_name.to_string()))
  };

  // Helper: build Integrate[pdf, {var, a, b}].
  let build_integral = |a: Expr, b: Expr| -> Expr {
    Expr::FunctionCall {
      name: "Integrate".to_string(),
      args: vec![
        pdf.clone(),
        Expr::List(vec![Expr::Identifier(var_name.to_string()), a, b].into()),
      ]
      .into(),
    }
  };

  // Conditional events come in as Conditioned[lhs, rhs] (now parsed natively).
  // P[A \[Conditioned] B, x ~ d] = P[A ∧ B, x ~ d] / P[B, x ~ d] —
  // handled by the caller; nothing to do here.

  // `And[a, b]`: intersection of two sub-ranges, i.e. the tighter range. We
  // detect simple `var > k1 && var > k2` (etc.) shapes that reduce to a
  // single bounded integral.
  if let Expr::FunctionCall {
    name: head,
    args: aargs,
  } = event
    && head == "And"
    && aargs.len() == 2
  {
    let extract = |e: &Expr| -> Option<(ComparisonOp, Expr)> {
      // Returns (op, threshold) for `var <op> k` or `k <op> var` rephrased
      // so the variable is on the left.
      if let Expr::Comparison {
        operands,
        operators,
      } = e
        && operands.len() == 2
        && operators.len() == 1
      {
        let op = operators[0];
        let left = &operands[0];
        let right = &operands[1];
        let is_left_var = matches!(left, Expr::Identifier(n) if n == var_name);
        let is_right_var =
          matches!(right, Expr::Identifier(n) if n == var_name);
        if is_left_var {
          return Some((op, right.clone()));
        }
        if is_right_var {
          let flipped = match op {
            ComparisonOp::Less => ComparisonOp::Greater,
            ComparisonOp::LessEqual => ComparisonOp::GreaterEqual,
            ComparisonOp::Greater => ComparisonOp::Less,
            ComparisonOp::GreaterEqual => ComparisonOp::LessEqual,
            other => other,
          };
          return Some((flipped, left.clone()));
        }
      }
      None
    };
    if let (Some((op1, k1)), Some((op2, k2))) =
      (extract(&aargs[0]), extract(&aargs[1]))
    {
      let is_lower = |op: &ComparisonOp| {
        matches!(op, ComparisonOp::Greater | ComparisonOp::GreaterEqual)
      };
      let is_upper = |op: &ComparisonOp| {
        matches!(op, ComparisonOp::Less | ComparisonOp::LessEqual)
      };
      // Both lower bounds → use the larger one. Both upper → use the smaller.
      // One of each → bounded interval.
      let new_lo;
      let new_hi;
      if is_lower(&op1) && is_lower(&op2) {
        new_lo = Expr::FunctionCall {
          name: "Max".to_string(),
          args: vec![k1, k2].into(),
        };
        new_hi = hi.clone();
      } else if is_upper(&op1) && is_upper(&op2) {
        new_lo = lo.clone();
        new_hi = Expr::FunctionCall {
          name: "Min".to_string(),
          args: vec![k1, k2].into(),
        };
      } else if is_lower(&op1) && is_upper(&op2) {
        new_lo = k1;
        new_hi = k2;
      } else if is_upper(&op1) && is_lower(&op2) {
        new_lo = k2;
        new_hi = k1;
      } else {
        return Ok(None);
      }
      let integral = build_integral(new_lo, new_hi);
      return Ok(Some(eval(integral)?));
    }
  }

  // Two-operand comparison
  if let Expr::Comparison {
    operands,
    operators,
  } = event
    && operands.len() == 2
    && operators.len() == 1
  {
    let op = &operators[0];
    let left = &operands[0];
    let right = &operands[1];
    let is_left_var = matches!(left, Expr::Identifier(n) if n == var_name);
    let is_right_var = matches!(right, Expr::Identifier(n) if n == var_name);

    // x > k or x >= k  →  Integrate[pdf, {x, k, hi}]
    if is_left_var
      && (*op == ComparisonOp::Greater || *op == ComparisonOp::GreaterEqual)
    {
      let integral = build_integral(right.clone(), hi.clone());
      return Ok(Some(eval(integral)?));
    }
    // x < k or x <= k  →  Integrate[pdf, {x, lo, k}]
    if is_left_var
      && (*op == ComparisonOp::Less || *op == ComparisonOp::LessEqual)
    {
      let integral = build_integral(lo.clone(), right.clone());
      return Ok(Some(eval(integral)?));
    }
    // k < x → x > k
    if is_right_var
      && (*op == ComparisonOp::Less || *op == ComparisonOp::LessEqual)
    {
      let integral = build_integral(left.clone(), hi.clone());
      return Ok(Some(eval(integral)?));
    }
    // k > x → x < k
    if is_right_var
      && (*op == ComparisonOp::Greater || *op == ComparisonOp::GreaterEqual)
    {
      let integral = build_integral(lo.clone(), left.clone());
      return Ok(Some(eval(integral)?));
    }
  }

  // Three-operand chained comparison: a < x < b
  if let Expr::Comparison {
    operands,
    operators,
  } = event
    && operands.len() == 3
    && operators.len() == 2
    && matches!(&operands[1], Expr::Identifier(n) if n == var_name)
  {
    let both_less = matches!(
      (&operators[0], &operators[1]),
      (
        ComparisonOp::Less | ComparisonOp::LessEqual,
        ComparisonOp::Less | ComparisonOp::LessEqual
      )
    );
    if both_less {
      let integral = build_integral(operands[0].clone(), operands[2].clone());
      return Ok(Some(eval(integral)?));
    }
  }

  Ok(None)
}

/// Numerical wrapper for `Probability` — returns `N[Probability[…]]`.
pub fn n_probability_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let prob = probability_ast(args)?;
  // If it stayed unevaluated, re-wrap as NProbability so the user sees the
  // numerical-evaluation request rather than the symbolic fallback.
  if matches!(&prob, Expr::FunctionCall { name, .. } if name == "Probability") {
    return Ok(Expr::FunctionCall {
      name: "NProbability".to_string(),
      args: args.to_vec().into(),
    });
  }
  eval(Expr::FunctionCall {
    name: "N".to_string(),
    args: vec![prob].into(),
  })
}

/// Numerical wrapper for `Expectation` — returns `N[Expectation[…]]`.
pub fn n_expectation_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let exp = expectation_ast(args)?;
  if matches!(&exp, Expr::FunctionCall { name, .. } if name == "Expectation") {
    return Ok(Expr::FunctionCall {
      name: "NExpectation".to_string(),
      args: args.to_vec().into(),
    });
  }
  eval(Expr::FunctionCall {
    name: "N".to_string(),
    args: vec![exp].into(),
  })
}

/// Implements `Expectation[f(x), x ~ CensoredDistribution[{a, b}, base]]`.
/// `dargs` is `[{a, b}, base_dist]`. Returns `None` if the shape doesn't
/// match so the caller can fall through.
fn try_expectation_censored(
  expr: &Expr,
  var: &str,
  dargs: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  use crate::functions::plot::substitute_var;
  if dargs.len() != 2 {
    return Ok(None);
  }
  let Expr::List(bounds) = &dargs[0] else {
    return Ok(None);
  };
  if bounds.len() != 2 {
    return Ok(None);
  }
  let a = bounds[0].clone();
  let b = bounds[1].clone();
  let base = &dargs[1];

  // For a ProbabilityDistribution base, we have direct access to the pdf and
  // its domain. The integral over [a, b] uses the renamed pdf.
  if let Expr::FunctionCall {
    name: base_name,
    args: base_args,
  } = base
    && base_name == "ProbabilityDistribution"
    && base_args.len() == 2
    && let Expr::List(iter) = &base_args[1]
    && iter.len() == 3
    && let Expr::Identifier(dummy) = &iter[0]
  {
    let pdf = if dummy == var {
      base_args[0].clone()
    } else {
      substitute_var(&base_args[0], dummy, &Expr::Identifier(var.to_string()))
    };
    // Integrate f(x) * pdf(x) from a to b.
    let mid_integral = Expr::FunctionCall {
      name: "Integrate".to_string(),
      args: vec![
        times(expr.clone(), pdf.clone()),
        Expr::List(
          vec![Expr::Identifier(var.to_string()), a.clone(), b.clone()].into(),
        ),
      ]
      .into(),
    };
    // P[X < a] and P[X > b] via integrals of pdf over (lo, a) / (b, hi).
    let p_below = {
      let lo = iter[1].clone();
      Expr::FunctionCall {
        name: "Integrate".to_string(),
        args: vec![
          pdf.clone(),
          Expr::List(
            vec![Expr::Identifier(var.to_string()), lo, a.clone()].into(),
          ),
        ]
        .into(),
      }
    };
    let p_above = {
      let hi = iter[2].clone();
      Expr::FunctionCall {
        name: "Integrate".to_string(),
        args: vec![
          pdf,
          Expr::List(
            vec![Expr::Identifier(var.to_string()), b.clone(), hi].into(),
          ),
        ]
        .into(),
      }
    };
    let f_at_a = substitute_var(expr, var, &a);
    let f_at_b = substitute_var(expr, var, &b);
    let total = plus(
      times(f_at_a, p_below),
      plus(eval(mid_integral)?, times(f_at_b, p_above)),
    );
    return Ok(Some(eval(total)?));
  }

  // For a base distribution Woxi already understands, use its CDF / PDF.
  // We compute the components via integrals against PDF if available.
  let base_name = match base {
    Expr::FunctionCall { name, .. } => name.as_str(),
    _ => return Ok(None),
  };
  // Generic numerical fallback would go here. For now, only the
  // ProbabilityDistribution path is wired up since that covers the
  // example notebook; bail out otherwise.
  let _ = base_name;
  Ok(None)
}

/// Compute `Expectation[f, vars \[Distributed] ProbabilityDistribution[pdf, {a,…}, {b,…}, …]]`
/// by reducing to `Integrate[f * pdf, {a,…}, …]` over the declared domain.
///
/// `dargs` is the argument list of the `ProbabilityDistribution` head:
/// `[pdf_expr, {var1, lo, hi}, {var2, lo, hi}, …]`.
/// `vars` are the names supplied in the `Distributed[…]` LHS; they replace
/// the distribution's dummy variable names so the user can write
/// `Expectation[x, x ~ ProbabilityDistribution[2/y^3, {y, 1, ∞}]]`
/// and have `y` rebound to `x` for the integral.
fn try_expectation_probability_distribution(
  expr: &Expr,
  vars: &[String],
  dargs: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  use crate::functions::plot::substitute_var;

  if dargs.len() < 2 {
    return Ok(None);
  }
  let pdf = &dargs[0];
  let iters = &dargs[1..];

  // Number of supplied variables must match the number of iterators.
  if iters.len() != vars.len() {
    return Ok(None);
  }

  // Substitute distribution's dummy variable names with user-supplied names
  // in both the pdf and integration iterators.
  let mut pdf_sub = pdf.clone();
  let mut new_iters: Vec<Expr> = Vec::with_capacity(iters.len());
  for (i, iter) in iters.iter().enumerate() {
    let Expr::List(items) = iter else {
      return Ok(None);
    };
    if items.len() != 3 {
      return Ok(None);
    }
    let Expr::Identifier(dummy) = &items[0] else {
      return Ok(None);
    };
    let target = &vars[i];
    if dummy != target {
      let new_var = Expr::Identifier(target.clone());
      pdf_sub = substitute_var(&pdf_sub, dummy, &new_var);
    }
    new_iters.push(Expr::List(
      vec![
        Expr::Identifier(target.clone()),
        items[1].clone(),
        items[2].clone(),
      ]
      .into(),
    ));
  }

  // Build Integrate[expr * pdf, iter1, iter2, …]
  let integrand = times(expr.clone(), pdf_sub);
  let mut integrate_args: Vec<Expr> = Vec::with_capacity(1 + new_iters.len());
  integrate_args.push(integrand);
  integrate_args.extend(new_iters);
  let integral = Expr::FunctionCall {
    name: "Integrate".to_string(),
    args: integrate_args.into(),
  };
  Ok(Some(eval(integral)?))
}

/// Public wrapper for distribution_mean_variance for use by Mean/Variance
pub fn distribution_mean_variance_pub(
  dist_name: &str,
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  distribution_mean_variance(dist_name, dargs)
}

/// Returns (Mean, Variance) as symbolic expressions for known distributions.
fn distribution_mean_variance(
  dist_name: &str,
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  match dist_name {
    "NormalDistribution" => {
      let (mu, sigma) = match dargs.len() {
        0 => (int(0), int(1)),
        2 => (dargs[0].clone(), dargs[1].clone()),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "NormalDistribution expects 0 or 2 arguments".into(),
          ));
        }
      };
      // Mean = mu, Var = sigma^2
      Ok((mu, power(sigma, int(2))))
    }
    "UniformDistribution" => {
      let (a, b) = match dargs.len() {
        0 => (int(0), int(1)),
        1 => {
          if let Expr::List(bounds) = &dargs[0] {
            if bounds.len() == 2 {
              (bounds[0].clone(), bounds[1].clone())
            } else {
              return Err(InterpreterError::EvaluationError(
                "UniformDistribution: expected {min, max}".into(),
              ));
            }
          } else {
            return Err(InterpreterError::EvaluationError(
              "UniformDistribution: expected a list".into(),
            ));
          }
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "UniformDistribution expects 0 or 1 argument".into(),
          ));
        }
      };
      // Mean = (a+b)/2, Var = (b-a)^2/12
      let mean = divide(plus(a.clone(), b.clone()), int(2));
      let var = divide(power(minus(b, a), int(2)), int(12));
      Ok((mean, var))
    }
    "ExponentialDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "ExponentialDistribution expects 1 argument".into(),
        ));
      }
      let lambda = dargs[0].clone();
      // Mean = 1/lambda, Var = 1/lambda^2
      let mean = divide(int(1), lambda.clone());
      let var = divide(int(1), power(lambda, int(2)));
      Ok((mean, var))
    }
    "PoissonDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "PoissonDistribution expects 1 argument".into(),
        ));
      }
      let mu = dargs[0].clone();
      // Mean = mu, Var = mu
      Ok((mu.clone(), mu))
    }
    "BernoulliDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "BernoulliDistribution expects 1 argument".into(),
        ));
      }
      let p = dargs[0].clone();
      // Mean = p, Var = p(1-p)
      let var = times(p.clone(), minus(int(1), p.clone()));
      Ok((p, var))
    }
    "BinomialDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "BinomialDistribution expects 2 arguments".into(),
        ));
      }
      let n = dargs[0].clone();
      let p = dargs[1].clone();
      // Mean = n*p, Var = n*(1-p)*p
      let mean = times(n.clone(), p.clone());
      let var = times(times(n, minus(int(1), p.clone())), p);
      Ok((mean, var))
    }
    "GammaDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "GammaDistribution expects 2 arguments".into(),
        ));
      }
      let alpha = dargs[0].clone();
      let beta = dargs[1].clone();
      // Mean = alpha*beta, Var = alpha*beta^2
      let mean = times(alpha.clone(), beta.clone());
      let var = times(alpha, power(beta, int(2)));
      Ok((mean, var))
    }
    "BetaDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "BetaDistribution expects 2 arguments".into(),
        ));
      }
      let a = dargs[0].clone();
      let b = dargs[1].clone();
      // Mean = a/(a+b), Var = a*b / ((a+b)^2 * (a+b+1))
      let ab = plus(a.clone(), b.clone());
      let mean = divide(a.clone(), ab.clone());
      let var = divide(
        times(a, b),
        times(power(ab.clone(), int(2)), plus(ab, int(1))),
      );
      Ok((mean, var))
    }
    "StudentTDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "StudentTDistribution expects 1 argument".into(),
        ));
      }
      let nu = dargs[0].clone();
      // Mean = Piecewise[{{0, nu > 1}}, Indeterminate]
      let mean = piecewise(
        vec![(
          int(0),
          comparison(nu.clone(), ComparisonOp::Greater, int(1)),
        )],
        Expr::Identifier("Indeterminate".to_string()),
      );
      // Var = Piecewise[{{nu/(-2+nu), nu > 2}}, Indeterminate]
      let var = piecewise(
        vec![(
          divide(nu.clone(), plus(int(-2), nu.clone())),
          comparison(nu, ComparisonOp::Greater, int(2)),
        )],
        Expr::Identifier("Indeterminate".to_string()),
      );
      Ok((mean, var))
    }
    "LogNormalDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "LogNormalDistribution expects 2 arguments".into(),
        ));
      }
      let mu = dargs[0].clone();
      let sigma = dargs[1].clone();
      // Mean = E^(mu + sigma^2/2)
      let mean = power(
        Expr::Identifier("E".to_string()),
        plus(mu.clone(), divide(power(sigma.clone(), int(2)), int(2))),
      );
      // Var = E^(2*mu + sigma^2) * (E^(sigma^2) - 1)
      let var = times(
        power(
          Expr::Identifier("E".to_string()),
          plus(times(int(2), mu), power(sigma.clone(), int(2))),
        ),
        minus(
          power(Expr::Identifier("E".to_string()), power(sigma, int(2))),
          int(1),
        ),
      );
      Ok((mean, var))
    }
    "ChiSquareDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "ChiSquareDistribution expects 1 argument".into(),
        ));
      }
      let k = dargs[0].clone();
      // Mean = k, Var = 2*k
      let mean = k.clone();
      let var = times(int(2), k);
      Ok((mean, var))
    }
    "ParetoDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "ParetoDistribution expects 2 arguments".into(),
        ));
      }
      let k = dargs[0].clone();
      let a = dargs[1].clone();
      // Mean = Piecewise[{{a*k/(-1+a), a > 1}}, Indeterminate]
      let mean = piecewise(
        vec![(
          divide(times(a.clone(), k.clone()), plus(int(-1), a.clone())),
          comparison(a.clone(), ComparisonOp::Greater, int(1)),
        )],
        Expr::Identifier("Indeterminate".to_string()),
      );
      // Var = Piecewise[{{a*k^2 / ((-2+a)*(-1+a)^2), a > 2}}, Indeterminate]
      let var = piecewise(
        vec![(
          divide(
            times(a.clone(), power(k, int(2))),
            times(
              plus(int(-2), a.clone()),
              power(plus(int(-1), a.clone()), int(2)),
            ),
          ),
          comparison(a, ComparisonOp::Greater, int(2)),
        )],
        Expr::Identifier("Indeterminate".to_string()),
      );
      Ok((mean, var))
    }
    "WeibullDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "WeibullDistribution expects 2 arguments".into(),
        ));
      }
      let a = dargs[0].clone();
      let b = dargs[1].clone();
      // Mean = b * Gamma[1 + 1/a]
      let mean = times(
        b.clone(),
        Expr::FunctionCall {
          name: "Gamma".to_string(),
          args: vec![plus(int(1), divide(int(1), a.clone()))].into(),
        },
      );
      // Var = b^2 * (Gamma[1 + 2/a] - Gamma[1 + 1/a]^2)
      let var = times(
        power(b, int(2)),
        minus(
          Expr::FunctionCall {
            name: "Gamma".to_string(),
            args: vec![plus(int(1), divide(int(2), a.clone()))].into(),
          },
          power(
            Expr::FunctionCall {
              name: "Gamma".to_string(),
              args: vec![plus(int(1), divide(int(1), a))].into(),
            },
            int(2),
          ),
        ),
      );
      Ok((mean, var))
    }
    "HalfNormalDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "HalfNormalDistribution expects 1 argument".into(),
        ));
      }
      let theta = dargs[0].clone();
      // Mean = 1/theta
      let mean = divide(int(1), theta.clone());
      // Variance = (Pi - 2) / (2 * theta^2). Built as (-2 + Pi)/(...) so
      // the rendered output matches wolframscript's canonical ordering.
      let var =
        divide(plus(int(-2), pi()), times(int(2), power(theta, int(2))));
      Ok((mean, var))
    }
    "InverseGaussianDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "InverseGaussianDistribution expects 2 arguments".into(),
        ));
      }
      let mu = dargs[0].clone();
      let lambda = dargs[1].clone();
      // Mean = mu; Variance = mu^3 / lambda
      let mean = mu.clone();
      let var = divide(power(mu, int(3)), lambda);
      Ok((mean, var))
    }
    "LogisticDistribution" => {
      let (mu, beta) = match dargs.len() {
        0 => (int(0), int(1)),
        2 => (dargs[0].clone(), dargs[1].clone()),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "LogisticDistribution expects 0 or 2 arguments".into(),
          ));
        }
      };
      // Mean = mu; Variance = beta^2 * Pi^2 / 3.
      let mean = mu;
      let var = divide(times(power(beta, int(2)), power(pi(), int(2))), int(3));
      Ok((mean, var))
    }
    "HypoexponentialDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "HypoexponentialDistribution expects 1 argument".into(),
        ));
      }
      let Expr::List(rates) = &dargs[0] else {
        return Err(InterpreterError::EvaluationError(
          "HypoexponentialDistribution: expected a list of rates".into(),
        ));
      };
      if rates.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "HypoexponentialDistribution: rate list cannot be empty".into(),
        ));
      }
      // Mean = sum_i 1/lambda_i; Variance = sum_i 1/lambda_i^2. The
      // hypoexponential is a sum of independent exponentials, so the
      // mean and variance add term by term.
      let mean_terms: Vec<Expr> =
        rates.iter().map(|r| divide(int(1), r.clone())).collect();
      let var_terms: Vec<Expr> = rates
        .iter()
        .map(|r| divide(int(1), power(r.clone(), int(2))))
        .collect();
      let mean = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: mean_terms.into(),
      };
      let var = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: var_terms.into(),
      };
      Ok((mean, var))
    }
    "ExtremeValueDistribution" => {
      let (a, b) = match dargs.len() {
        0 => (int(0), int(1)),
        2 => (dargs[0].clone(), dargs[1].clone()),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "ExtremeValueDistribution expects 0 or 2 arguments".into(),
          ));
        }
      };
      // Mean = a + b * EulerGamma
      let mean = plus(
        a,
        times(b.clone(), Expr::Constant("EulerGamma".to_string())),
      );
      // Variance = b^2 * Pi^2 / 6
      let var = divide(times(power(b, int(2)), power(pi(), int(2))), int(6));
      Ok((mean, var))
    }
    "StableDistribution" => {
      // Canonical 5-arg form: StableDistribution[type, alpha, beta, mu, sigma].
      // type ∈ {0, 1} selects between the two standard parametrisations.
      if dargs.len() != 5 {
        return Err(InterpreterError::EvaluationError(
          "StableDistribution expects 5 arguments (canonical form)".into(),
        ));
      }
      let type_ = dargs[0].clone();
      let alpha = dargs[1].clone();
      let beta = dargs[2].clone();
      let mu = dargs[3].clone();
      let sigma = dargs[4].clone();
      let indet = Expr::Identifier("Indeterminate".to_string());
      // Mean exists when 1 < alpha <= 2.
      // Type 0: Mean = mu - beta * sigma * Tan[Pi * alpha / 2]
      // Type 1: Mean = mu
      let mean_branch = match &type_ {
        Expr::Integer(0) => {
          let tan_arg = divide(times(alpha.clone(), pi()), int(2));
          let tan_term = Expr::FunctionCall {
            name: "Tan".to_string(),
            args: vec![tan_arg].into(),
          };
          minus(mu.clone(), times(times(beta, sigma.clone()), tan_term))
        }
        _ => mu.clone(),
      };
      let mean = piecewise(
        vec![(
          mean_branch,
          comparison3(
            int(2),
            ComparisonOp::GreaterEqual,
            alpha.clone(),
            ComparisonOp::Greater,
            int(1),
          ),
        )],
        indet.clone(),
      );
      // Variance is finite only for alpha == 2 (Gaussian limit), where it
      // is 2 * sigma^2. Holds for both type parametrisations.
      let var = piecewise(
        vec![(
          times(int(2), power(sigma, int(2))),
          comparison(alpha, ComparisonOp::Equal, int(2)),
        )],
        indet,
      );
      Ok((mean, var))
    }
    "InverseGammaDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "InverseGammaDistribution expects 2 arguments".into(),
        ));
      }
      let a = dargs[0].clone();
      let b = dargs[1].clone();
      let indet = Expr::Identifier("Indeterminate".to_string());
      // Mean = Piecewise[{{b/(-1 + a), a > 1}}, Indeterminate]
      let mean_branch = divide(b.clone(), plus(int(-1), a.clone()));
      let mean = piecewise(
        vec![(
          mean_branch,
          comparison(a.clone(), ComparisonOp::Greater, int(1)),
        )],
        indet.clone(),
      );
      // Variance = Piecewise[{{b^2/((-2 + a)*(-1 + a)^2), a > 2}},
      //                      Indeterminate]
      let denom = times(
        plus(int(-2), a.clone()),
        power(plus(int(-1), a.clone()), int(2)),
      );
      let var_branch = divide(power(b, int(2)), denom);
      let var = piecewise(
        vec![(var_branch, comparison(a, ComparisonOp::Greater, int(2)))],
        indet,
      );
      Ok((mean, var))
    }
    "GompertzMakehamDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "GompertzMakehamDistribution expects 2 arguments".into(),
        ));
      }
      let lambda = dargs[0].clone();
      let xi = dargs[1].clone();
      // Mean = (E^xi * Gamma[0, xi]) / lambda
      let gamma_0_xi = Expr::FunctionCall {
        name: "Gamma".to_string(),
        args: vec![int(0), xi.clone()].into(),
      };
      let mean = divide(times(power(e(), xi), gamma_0_xi), lambda.clone());
      // Variance has no simple closed form in elementary functions;
      // GompertzMakehamDistribution is intentionally absent from the
      // Variance dispatch list, so this placeholder is never returned
      // to the user. Provide an unevaluated stub so the tuple typechecks.
      let var = Expr::FunctionCall {
        name: "Variance".to_string(),
        args: vec![Expr::FunctionCall {
          name: "GompertzMakehamDistribution".to_string(),
          args: dargs.to_vec().into(),
        }]
        .into(),
      };
      Ok((mean, var))
    }
    "FrechetDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "FrechetDistribution expects 2 arguments".into(),
        ));
      }
      let a = dargs[0].clone();
      let b = dargs[1].clone();
      // Mean = Piecewise[{{b * Gamma[1 - 1/a], 1 < a}}, Infinity]
      let gamma_1_minus_inv_a = Expr::FunctionCall {
        name: "Gamma".to_string(),
        args: vec![minus(int(1), divide(int(1), a.clone()))].into(),
      };
      let mean = piecewise(
        vec![(
          times(b.clone(), gamma_1_minus_inv_a.clone()),
          comparison(int(1), ComparisonOp::Less, a.clone()),
        )],
        Expr::Identifier("Infinity".to_string()),
      );
      // Var = Piecewise[{{b^2 * (Gamma[1 - 2/a] - Gamma[1 - 1/a]^2), a > 2}}, Infinity]
      let gamma_1_minus_2_a = Expr::FunctionCall {
        name: "Gamma".to_string(),
        args: vec![minus(int(1), divide(int(2), a.clone()))].into(),
      };
      let var = piecewise(
        vec![(
          times(
            power(b, int(2)),
            minus(gamma_1_minus_2_a, power(gamma_1_minus_inv_a, int(2))),
          ),
          comparison(a, ComparisonOp::Greater, int(2)),
        )],
        Expr::Identifier("Infinity".to_string()),
      );
      Ok((mean, var))
    }
    "GeometricDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "GeometricDistribution expects 1 argument".into(),
        ));
      }
      let p = dargs[0].clone();
      // Mean = (1-p)/p, Var = (1-p)/p^2
      let one_minus_p = minus(int(1), p.clone());
      let mean = divide(one_minus_p.clone(), p.clone());
      let var = divide(one_minus_p, power(p, int(2)));
      Ok((mean, var))
    }
    "CauchyDistribution" => {
      // Mean and Variance are both Indeterminate for Cauchy
      let indet = Expr::Identifier("Indeterminate".to_string());
      Ok((indet.clone(), indet))
    }
    "LaplaceDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "LaplaceDistribution expects 2 arguments".into(),
        ));
      }
      let mu = dargs[0].clone();
      let b = dargs[1].clone();
      // Mean = mu, Var = 2*b^2
      let var = times(int(2), power(b, int(2)));
      Ok((mu, var))
    }
    "RayleighDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "RayleighDistribution expects 1 argument".into(),
        ));
      }
      let s = dargs[0].clone();
      // Mean = Sqrt[Pi/2] * s
      let mean = times(sqrt(divide(pi(), int(2))), s.clone());
      // Var = (2 - Pi/2) * s^2
      let var = times(minus(int(2), divide(pi(), int(2))), power(s, int(2)));
      Ok((mean, var))
    }
    "DiscreteUniformDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "DiscreteUniformDistribution expects 1 argument".into(),
        ));
      }
      let (imin, imax) = match &dargs[0] {
        Expr::List(bounds) if bounds.len() == 2 => {
          (bounds[0].clone(), bounds[1].clone())
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "DiscreteUniformDistribution expects a list {imin, imax}".into(),
          ));
        }
      };
      // Mean = (imin + imax) / 2
      let mean = divide(plus(imin.clone(), imax.clone()), int(2));
      // Variance = ((imax - imin) * (imax - imin + 2)) / 12
      let diff = minus(imax, imin);
      let var = divide(times(diff.clone(), plus(diff, int(2))), int(12));
      Ok((mean, var))
    }
    "NegativeBinomialDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "NegativeBinomialDistribution expects 2 arguments".into(),
        ));
      }
      let n = dargs[0].clone();
      let p = dargs[1].clone();
      // Mean = n*(1-p)/p, Var = n*(1-p)/p^2
      // Express variance as Times[n, (1-p), p^(-2)] so Sqrt can extract p^(-1)
      let one_minus_p = minus(int(1), p.clone());
      let mean = divide(times(n.clone(), one_minus_p.clone()), p.clone());
      let var = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![n, one_minus_p, power(p, int(-2))].into(),
      };
      Ok((mean, var))
    }
    "PascalDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "PascalDistribution expects 2 arguments".into(),
        ));
      }
      let n = dargs[0].clone();
      let p = dargs[1].clone();
      // Mean = n/p, Var = n*(1-p)/p^2
      let mean = divide(n.clone(), p.clone());
      let one_minus_p = minus(int(1), p.clone());
      let var = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![n, one_minus_p, power(p, int(-2))].into(),
      };
      Ok((mean, var))
    }
    "DagumDistribution" => {
      if dargs.len() != 3 {
        return Err(InterpreterError::EvaluationError(
          "DagumDistribution expects 3 arguments".into(),
        ));
      }
      let p = dargs[0].clone();
      let a = dargs[1].clone();
      let b = dargs[2].clone();
      // Mean = b * Gamma[(-1 + a)/a] * Gamma[1/a + p] / Gamma[p]
      let gamma = |x: Expr| Expr::FunctionCall {
        name: "Gamma".to_string(),
        args: vec![x].into(),
      };
      let mean = times(
        times(
          b.clone(),
          gamma(divide(minus(a.clone(), int(1)), a.clone())),
        ),
        divide(
          gamma(plus(divide(int(1), a.clone()), p.clone())),
          gamma(p.clone()),
        ),
      );
      // Var = -Mean^2 + b^2 * Gamma[(-2+a)/a] * Gamma[2/a + p] / Gamma[p]
      let second_moment = times(
        times(
          power(b, int(2)),
          gamma(divide(minus(a.clone(), int(2)), a.clone())),
        ),
        divide(gamma(plus(divide(int(2), a), p.clone())), gamma(p)),
      );
      let var = minus(second_moment, power(mean.clone(), int(2)));
      Ok((mean, var))
    }
    "NoncentralFRatioDistribution" => {
      if dargs.len() != 3 {
        return Err(InterpreterError::EvaluationError(
          "NoncentralFRatioDistribution expects 3 arguments".into(),
        ));
      }
      let n = dargs[0].clone();
      let m = dargs[1].clone();
      let l = dargs[2].clone();

      // Mean = Piecewise[{{m*(l+n)/((m-2)*n), m > 2}}, Indeterminate]
      let mean_expr = divide(
        times(m.clone(), plus(l.clone(), n.clone())),
        times(plus(int(-2), m.clone()), n.clone()),
      );
      let mean = piecewise(
        vec![(
          mean_expr,
          comparison(m.clone(), ComparisonOp::Greater, int(2)),
        )],
        Expr::Identifier("Indeterminate".to_string()),
      );

      // Variance = Piecewise[{{2*m^2*((l+n)^2 + (m-2)*(2*l+n)) / ((m-4)*(m-2)^2*n^2), m > 4}}, Indeterminate]
      let l_plus_n = plus(l.clone(), n.clone());
      let var_num = times(
        times(int(2), power(m.clone(), int(2))),
        plus(
          power(l_plus_n, int(2)),
          times(plus(int(-2), m.clone()), plus(times(int(2), l), n.clone())),
        ),
      );
      let var_den = times(
        times(
          plus(int(-4), m.clone()),
          power(plus(int(-2), m.clone()), int(2)),
        ),
        power(n, int(2)),
      );
      let var_expr = divide(var_num, var_den);
      let var = piecewise(
        vec![(var_expr, comparison(m, ComparisonOp::Greater, int(4)))],
        Expr::Identifier("Indeterminate".to_string()),
      );

      Ok((mean, var))
    }
    "HyperbolicDistribution" => {
      if dargs.len() != 4 {
        return Err(InterpreterError::EvaluationError(
          "HyperbolicDistribution expects 4 arguments".into(),
        ));
      }
      let a = dargs[0].clone();
      let b = dargs[1].clone();
      let d = dargs[2].clone();
      let m = dargs[3].clone();

      let besselk = |n: Expr, z: Expr| Expr::FunctionCall {
        name: "BesselK".to_string(),
        args: vec![n, z].into(),
      };

      let a2_minus_b2 =
        minus(power(a.clone(), int(2)), power(b.clone(), int(2)));
      let sqrt_a2_minus_b2 = sqrt(a2_minus_b2.clone());
      let k_arg = times(sqrt_a2_minus_b2.clone(), d.clone());
      let bk1 = besselk(int(1), k_arg.clone());
      let bk2 = besselk(int(2), k_arg.clone());
      let bk3 = besselk(int(3), k_arg);

      // Mean = m + b*d*BesselK[2, Sqrt[a^2-b^2]*d] / (Sqrt[a^2-b^2]*BesselK[1, Sqrt[a^2-b^2]*d])
      let mean = plus(
        m,
        divide(
          times(times(b.clone(), d.clone()), bk2.clone()),
          times(sqrt_a2_minus_b2, bk1.clone()),
        ),
      );

      // Variance = d*BesselK[2,z]/(sqrt_ab*BesselK[1,z])
      //          - b^2*d^2*BesselK[2,z]^2/((a^2-b^2)*BesselK[1,z]^2)
      //          + b^2*d^2*BesselK[3,z]/((a^2-b^2)*BesselK[1,z])
      let sqrt_ab = sqrt(a2_minus_b2.clone());
      let b2d2 = times(power(b, int(2)), power(d.clone(), int(2)));
      let term1 = divide(times(d, bk2.clone()), times(sqrt_ab, bk1.clone()));
      let term2 = divide(
        times(b2d2.clone(), power(bk2, int(2))),
        times(a2_minus_b2.clone(), power(bk1.clone(), int(2))),
      );
      let term3 = divide(times(b2d2, bk3), times(a2_minus_b2, bk1));
      let var = plus(minus(term1, term2), term3);
      Ok((mean, var))
    }
    "JohnsonDistribution" => {
      if dargs.len() != 5 {
        return Err(InterpreterError::EvaluationError(
          "JohnsonDistribution expects 5 arguments".into(),
        ));
      }
      let type_str = match &dargs[0] {
        Expr::String(s) => s.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "JohnsonDistribution: first argument must be a string type".into(),
          ));
        }
      };
      let gamma = dargs[1].clone();
      let delta = dargs[2].clone();
      let mu = dargs[3].clone();
      let sigma = dargs[4].clone();
      match type_str.as_str() {
        "SN" => {
          // Mean = (delta*mu - gamma*sigma)/delta
          let mean = divide(
            minus(
              times(delta.clone(), mu),
              times(gamma.clone(), sigma.clone()),
            ),
            delta.clone(),
          );
          // Var = sigma^2/delta^2
          let var = divide(power(sigma, int(2)), power(delta, int(2)));
          Ok((mean, var))
        }
        "SL" => {
          // Mean = mu + E^((1 - 2*delta*gamma)/(2*delta^2)) * sigma
          let exp_arg = divide(
            minus(int(1), times(int(2), times(delta.clone(), gamma.clone()))),
            times(int(2), power(delta.clone(), int(2))),
          );
          let mean = plus(mu, times(power(e(), exp_arg), sigma.clone()));
          // Var = E^((1 - 2*delta*gamma)/delta^2) * (-1 + E^(1/delta^2)) * sigma^2
          let exp_arg2 = divide(
            minus(int(1), times(int(2), times(delta.clone(), gamma))),
            power(delta.clone(), int(2)),
          );
          let inv_delta_sq = divide(int(1), power(delta.clone(), int(2)));
          let var = times(
            times(
              power(e(), exp_arg2),
              minus(power(e(), inv_delta_sq), int(1)),
            ),
            power(sigma, int(2)),
          );
          Ok((mean, var))
        }
        "SU" => {
          // Mean = mu - sigma * E^(1/(2*delta^2)) * Sinh[gamma/delta]
          // (Wolfram expands Sinh differently, so this is added to skip list)
          let delta_sq = power(delta.clone(), int(2));
          let exp_half =
            power(e(), divide(int(1), times(int(2), delta_sq.clone())));
          let sinh_gd = Expr::FunctionCall {
            name: "Sinh".to_string(),
            args: vec![divide(gamma.clone(), delta.clone())].into(),
          };
          let mean = minus(mu, times(sigma.clone(), times(exp_half, sinh_gd)));
          // Var = (sigma^2/2) * (Exp[1/delta^2] - 1) * (Exp[1/delta^2]*Cosh[2*gamma/delta] + 1)
          let exp_full = power(e(), divide(int(1), delta_sq));
          let cosh_2gd = Expr::FunctionCall {
            name: "Cosh".to_string(),
            args: vec![divide(times(int(2), gamma), delta)].into(),
          };
          let var = times(
            divide(power(sigma, int(2)), int(2)),
            times(
              minus(exp_full.clone(), int(1)),
              plus(times(exp_full, cosh_2gd), int(1)),
            ),
          );
          Ok((mean, var))
        }
        "SB" => {
          // Mean for SB doesn't have a simple closed form
          Err(InterpreterError::EvaluationError(
            "JohnsonDistribution SB: no closed-form mean/variance".into(),
          ))
        }
        _ => Err(InterpreterError::EvaluationError(format!(
          "JohnsonDistribution: unknown type {type_str}"
        ))),
      }
    }
    "ArcSinDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "ArcSinDistribution expects 1 argument (a list {a, b})".into(),
        ));
      }
      let (a, b) = match &dargs[0] {
        Expr::List(bounds) if bounds.len() == 2 => {
          (bounds[0].clone(), bounds[1].clone())
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "ArcSinDistribution expects a list {a, b}".into(),
          ));
        }
      };
      // Mean = (a + b) / 2
      let mean = divide(plus(a.clone(), b.clone()), int(2));
      // Variance = (b - a)^2 / 8
      let var = divide(power(minus(b, a), int(2)), int(8));
      Ok((mean, var))
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "Expectation: unsupported distribution {dist_name}"
    ))),
  }
}

/// Check if expression is var^n
fn is_power_of_var(expr: &Expr, var: &str, n: i128) -> bool {
  match expr {
    Expr::BinaryOp { op, left, right }
      if *op == crate::syntax::BinaryOperator::Power =>
    {
      matches!(left.as_ref(), Expr::Identifier(v) if v == var)
        && matches!(right.as_ref(), Expr::Integer(k) if *k == n)
    }
    _ => false,
  }
}

/// Try to extract a linear expression a*x + b from expr.
/// Returns Some((a, b)) if expr is linear in var.
fn extract_linear(expr: &Expr, var: &str) -> Option<(Expr, Expr)> {
  match expr {
    // Pure variable: x → (1, 0)
    Expr::Identifier(n) if n == var => Some((int(1), int(0))),
    // a * x
    Expr::BinaryOp { op, left, right }
      if *op == crate::syntax::BinaryOperator::Times =>
    {
      if matches!(right.as_ref(), Expr::Identifier(n) if n == var)
        && !contains_var(left, var)
      {
        Some((left.as_ref().clone(), int(0)))
      } else if matches!(left.as_ref(), Expr::Identifier(n) if n == var)
        && !contains_var(right, var)
      {
        Some((right.as_ref().clone(), int(0)))
      } else {
        None
      }
    }
    // a + b (try linear decomposition)
    Expr::BinaryOp { op, left, right }
      if *op == crate::syntax::BinaryOperator::Plus =>
    {
      // If left is linear in var and right is constant (or vice versa)
      if !contains_var(right, var)
        && let Some((a, b)) = extract_linear(left, var)
      {
        let new_b = plus(b, right.as_ref().clone());
        return Some((a, new_b));
      }
      if !contains_var(left, var)
        && let Some((a, b)) = extract_linear(right, var)
      {
        let new_b = plus(left.as_ref().clone(), b);
        return Some((a, new_b));
      }
      None
    }
    Expr::BinaryOp { op, left, right }
      if *op == crate::syntax::BinaryOperator::Minus =>
    {
      if !contains_var(right, var)
        && let Some((a, b)) = extract_linear(left, var)
      {
        let new_b = minus(b, right.as_ref().clone());
        return Some((a, new_b));
      }
      None
    }
    // FunctionCall Times[...] form
    Expr::FunctionCall { name, args } if name == "Times" => {
      // Find which arg is the variable and which are constants
      let mut var_idx = None;
      for (i, arg) in args.iter().enumerate() {
        if contains_var(arg, var) {
          if var_idx.is_some() {
            return None; // multiple args contain var
          }
          var_idx = Some(i);
        }
      }
      if let Some(vi) = var_idx {
        if !matches!(&args[vi], Expr::Identifier(n) if n == var) {
          return None;
        }
        let mut coeff_parts: Vec<Expr> = Vec::new();
        for (i, arg) in args.iter().enumerate() {
          if i != vi {
            coeff_parts.push(arg.clone());
          }
        }
        let coeff = if coeff_parts.len() == 1 {
          coeff_parts.pop().unwrap()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: coeff_parts.into(),
          }
        };
        Some((coeff, int(0)))
      } else {
        None
      }
    }
    // FunctionCall Plus[...] form
    Expr::FunctionCall { name, args } if name == "Plus" => {
      let mut total_a = int(0);
      let mut total_b = int(0);
      for arg in args {
        if contains_var(arg, var) {
          if let Some((a, b)) = extract_linear(arg, var) {
            total_a = plus(total_a, a);
            total_b = plus(total_b, b);
          } else {
            return None;
          }
        } else {
          total_b = plus(total_b, arg.clone());
        }
      }
      Some((total_a, total_b))
    }
    // Constant (no var)
    _ if !contains_var(expr, var) => None, // Not linear, it's constant - caller handles
    _ => None,
  }
}

fn contains_var(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::Identifier(n) => n == var,
    Expr::Integer(_) | Expr::Real(_) | Expr::String(_) => false,
    Expr::BinaryOp { left, right, .. } => {
      contains_var(left, var) || contains_var(right, var)
    }
    Expr::UnaryOp { operand, .. } => contains_var(operand, var),
    Expr::FunctionCall { name, args } => {
      if name == "Rational" && args.len() == 2 {
        return false; // Rational[n, d] is a constant
      }
      args.iter().any(|a| contains_var(a, var))
    }
    Expr::List(items) => items.iter().any(|a| contains_var(a, var)),
    _ => true, // conservative
  }
}

/// Numerical expectation via Monte Carlo or numerical integration.
fn expectation_numerical(
  expr: &Expr,
  var: &str,
  dist_name: &str,
  dargs: &[Expr],
) -> Result<Expr, InterpreterError> {
  use crate::functions::plot::substitute_var;

  // Get integration range and PDF for quadrature
  let n_points = 1000;

  // Determine integration bounds based on distribution
  let (lo, hi): (f64, f64) = match dist_name {
    "UniformDistribution" => {
      let (a, b) = match dargs.len() {
        0 => (0.0, 1.0),
        1 => {
          if let Expr::List(bounds) = &dargs[0] {
            if bounds.len() == 2 {
              let a = try_eval_to_f64(&bounds[0]).unwrap_or(0.0);
              let b = try_eval_to_f64(&bounds[1]).unwrap_or(1.0);
              (a, b)
            } else {
              (0.0, 1.0)
            }
          } else {
            (0.0, 1.0)
          }
        }
        _ => (0.0, 1.0),
      };
      (a, b)
    }
    "ExponentialDistribution" => {
      // Integrate from 0 to ~10/lambda
      let lambda = try_eval_to_f64(&dargs[0]).unwrap_or(1.0);
      (0.0, 10.0 / lambda)
    }
    "NormalDistribution" => {
      let (mu, sigma) = match dargs.len() {
        0 => (0.0, 1.0),
        2 => {
          let m = try_eval_to_f64(&dargs[0]).unwrap_or(0.0);
          let s = try_eval_to_f64(&dargs[1]).unwrap_or(1.0);
          (m, s)
        }
        _ => (0.0, 1.0),
      };
      (mu - 6.0 * sigma, mu + 6.0 * sigma)
    }
    _ => {
      // Return unevaluated for unsupported distributions
      return Ok(Expr::FunctionCall {
        name: "Expectation".to_string(),
        args: vec![
          expr.clone(),
          Expr::FunctionCall {
            name: "Distributed".to_string(),
            args: vec![
              Expr::Identifier(var.to_string()),
              Expr::FunctionCall {
                name: dist_name.to_string(),
                args: dargs.to_vec().into(),
              },
            ]
            .into(),
          },
        ]
        .into(),
      });
    }
  };

  // Numerical integration: E[f(x)] = integral f(x) * pdf(x) dx
  let dx = (hi - lo) / n_points as f64;
  let mut sum = 0.0;
  let dist_expr = Expr::FunctionCall {
    name: dist_name.to_string(),
    args: dargs.to_vec().into(),
  };

  for i in 0..=n_points {
    let x = lo + i as f64 * dx;
    let x_expr = Expr::Real(x);

    // Evaluate f(x)
    let fx_sub = substitute_var(expr, var, &x_expr);
    let fx_val = evaluate_expr_to_expr(&fx_sub)
      .ok()
      .and_then(|e| try_eval_to_f64(&e))
      .unwrap_or(0.0);

    // Evaluate pdf(x)
    let pdf_val = pdf_ast(&[dist_expr.clone(), x_expr])
      .ok()
      .and_then(|e| try_eval_to_f64(&e))
      .unwrap_or(0.0);

    let weight = if i == 0 || i == n_points { 0.5 } else { 1.0 };
    sum += weight * fx_val * pdf_val;
  }
  sum *= dx;

  // Round to reasonable precision
  let result = (sum * 1e10).round() / 1e10;
  if (result - result.round()).abs() < 1e-8 {
    Ok(Expr::Integer(result.round() as i128))
  } else {
    Ok(Expr::Real(result))
  }
}

/// PDF[BetaDistribution[a, b], x] = Piecewise[{{x^(a-1) * (1-x)^(b-1) / Beta[a,b], 0 < x < 1}}, 0]
fn pdf_beta(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "BetaDistribution expects 2 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();

  // x^(a-1)
  let x_part = power(x.clone(), minus(a.clone(), int(1)));
  // (1-x)^(b-1)
  let one_minus_x_part =
    power(minus(int(1), x.clone()), minus(b.clone(), int(1)));
  // Beta[a, b]
  let beta_fn = Expr::FunctionCall {
    name: "Beta".to_string(),
    args: vec![a, b].into(),
  };
  let value = eval(divide(times(x_part, one_minus_x_part), beta_fn))?;
  let cond =
    comparison3(int(0), ComparisonOp::Less, x, ComparisonOp::Less, int(1));
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// CDF[BetaDistribution[a, b], x] = Piecewise[{{BetaRegularized[x, a, b], 0 < x < 1}, {1, x >= 1}}, 0]
fn cdf_beta(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "BetaDistribution expects 2 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();

  let value = Expr::FunctionCall {
    name: "BetaRegularized".to_string(),
    args: vec![x.clone(), a, b].into(),
  };
  let cond1 = comparison3(
    int(0),
    ComparisonOp::Less,
    x.clone(),
    ComparisonOp::Less,
    int(1),
  );
  let cond2 = comparison(x, ComparisonOp::GreaterEqual, int(1));
  eval(piecewise(vec![(value, cond1), (int(1), cond2)], int(0)))
}

/// CDF[StudentTDistribution[nu], x] = Piecewise[
///   {{BetaRegularized[nu/(nu + x^2), nu/2, 1/2]/2, x <= 0}},
///   (1 + BetaRegularized[x^2/(nu + x^2), 1/2, nu/2])/2]
fn cdf_student_t(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "StudentTDistribution expects 1 argument".into(),
    ));
  }
  let nu = dargs[0].clone();
  let half = divide(int(1), int(2));
  let nu_over_2 = divide(nu.clone(), int(2));
  let x_sq = power(x.clone(), int(2));
  let nu_plus_x_sq = plus(nu.clone(), x_sq.clone());
  // Left branch: BetaRegularized[nu/(nu + x^2), nu/2, 1/2] / 2
  let left_arg = divide(nu, nu_plus_x_sq.clone());
  let left_beta = Expr::FunctionCall {
    name: "BetaRegularized".to_string(),
    args: vec![left_arg, nu_over_2.clone(), half.clone()].into(),
  };
  let left_value = divide(left_beta, int(2));
  // Right branch: (1 + BetaRegularized[x^2/(nu + x^2), 1/2, nu/2]) / 2
  let right_arg = divide(x_sq, nu_plus_x_sq);
  let right_beta = Expr::FunctionCall {
    name: "BetaRegularized".to_string(),
    args: vec![right_arg, half, nu_over_2].into(),
  };
  let right_value = divide(plus(int(1), right_beta), int(2));
  let cond = comparison(x, ComparisonOp::LessEqual, int(0));
  eval(piecewise(vec![(left_value, cond)], right_value))
}

/// PDF[StudentTDistribution[nu], x] = (1 + x^2/nu)^(-(1+nu)/2) / (Sqrt[nu] * Beta[nu/2, 1/2])
fn pdf_student_t(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "StudentTDistribution expects 1 argument".into(),
    ));
  }
  let nu = dargs[0].clone();

  // (1 + x^2/nu)^(-(1+nu)/2)
  let inner = plus(int(1), divide(power(x, int(2)), nu.clone()));
  let exponent = divide(
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(plus(int(1), nu.clone())),
    },
    int(2),
  );
  let numerator = power(inner, exponent);
  // Sqrt[nu] * Beta[nu/2, 1/2]
  let denominator = times(
    sqrt(nu.clone()),
    Expr::FunctionCall {
      name: "Beta".to_string(),
      args: vec![divide(nu, int(2)), divide(int(1), int(2))].into(),
    },
  );
  eval(divide(numerator, denominator))
}

/// PDF[LogNormalDistribution[mu, sigma], x] = Piecewise[{{1/(E^((Log[x]-mu)^2/(2*sigma^2))*Sqrt[2*Pi]*sigma*x), x > 0}}, 0]
fn pdf_lognormal(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LogNormalDistribution expects 2 arguments".into(),
    ));
  }
  let mu = dargs[0].clone();
  let sigma = dargs[1].clone();

  // 1 / (E^((Log[x] - mu)^2 / (2*sigma^2)) * Sqrt[2*Pi] * sigma * x)
  let log_x = Expr::FunctionCall {
    name: "Log".to_string(),
    args: vec![x.clone()].into(),
  };
  let exponent = divide(
    power(minus(log_x, mu), int(2)),
    times(int(2), power(sigma.clone(), int(2))),
  );
  let denom = times(
    times(
      power(Expr::Identifier("E".to_string()), exponent),
      sqrt(times(int(2), Expr::Identifier("Pi".to_string()))),
    ),
    times(sigma, x.clone()),
  );
  let pdf_val = divide(int(1), denom);

  // Piecewise[{{pdf_val, x > 0}}, 0]
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(pdf_val, cond)], int(0)))
}

/// CDF[LogNormalDistribution[mu, sigma], x] = Piecewise[{{Erfc[-(Log[x]-mu)/(Sqrt[2]*sigma)]/2, x > 0}}, 0]
fn cdf_lognormal(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LogNormalDistribution expects 2 arguments".into(),
    ));
  }
  let mu = dargs[0].clone();
  let sigma = dargs[1].clone();

  // Erfc[-(Log[x] - mu) / (Sqrt[2] * sigma)] / 2
  let log_x = Expr::FunctionCall {
    name: "Log".to_string(),
    args: vec![x.clone()].into(),
  };
  let arg = Expr::UnaryOp {
    op: crate::syntax::UnaryOperator::Minus,
    operand: Box::new(divide(minus(log_x, mu), times(sqrt(int(2)), sigma))),
  };
  let cdf_val = divide(
    Expr::FunctionCall {
      name: "Erfc".to_string(),
      args: vec![arg].into(),
    },
    int(2),
  );

  // Piecewise[{{cdf_val, x > 0}}, 0]
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(cdf_val, cond)], int(0)))
}

/// PDF[ChiSquareDistribution[k], x] = Piecewise[{{x^(k/2-1) / (2^(k/2) * E^(x/2) * Gamma[k/2]), x > 0}}, 0]
fn pdf_chi_square(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ChiSquareDistribution expects 1 argument".into(),
    ));
  }
  let k = dargs[0].clone();

  // x^(k/2 - 1)
  let x_power = power(x.clone(), minus(divide(k.clone(), int(2)), int(1)));
  // 2^(k/2)
  let two_power = power(int(2), divide(k.clone(), int(2)));
  // E^(x/2)
  let exp_part = power(e(), divide(x.clone(), int(2)));
  // Gamma[k/2]
  let gamma_part = Expr::FunctionCall {
    name: "Gamma".to_string(),
    args: vec![divide(k, int(2))].into(),
  };
  let denom = times(times(two_power, exp_part), gamma_part);
  let pdf_val = divide(x_power, denom);

  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(pdf_val, cond)], int(0)))
}

/// CDF[ChiSquareDistribution[k], x] = Piecewise[{{GammaRegularized[k/2, 0, x/2], x > 0}}, 0]
fn cdf_chi_square(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ChiSquareDistribution expects 1 argument".into(),
    ));
  }
  let k = dargs[0].clone();

  let cdf_val = Expr::FunctionCall {
    name: "GammaRegularized".to_string(),
    args: vec![divide(k, int(2)), int(0), divide(x.clone(), int(2))].into(),
  };

  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(cdf_val, cond)], int(0)))
}

/// PDF[ParetoDistribution[k, a], x] = Piecewise[{{a*k^a*x^(-1-a), x >= k}}, 0]
fn pdf_pareto(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ParetoDistribution expects 2 arguments".into(),
    ));
  }
  let k = dargs[0].clone();
  let a = dargs[1].clone();

  // a * k^a * x^(-1-a)
  let pdf_val = times(
    times(a.clone(), power(k.clone(), a.clone())),
    power(
      x.clone(),
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(plus(int(1), a)),
      },
    ),
  );
  let cond = comparison(x, ComparisonOp::GreaterEqual, k);
  eval(piecewise(vec![(pdf_val, cond)], int(0)))
}

/// CDF[ParetoDistribution[k, a], x] = Piecewise[{{1 - (k/x)^a, x >= k}}, 0]
fn cdf_pareto(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ParetoDistribution expects 2 arguments".into(),
    ));
  }
  let k = dargs[0].clone();
  let a = dargs[1].clone();

  let cdf_val = minus(int(1), power(divide(k.clone(), x.clone()), a));
  let cond = comparison(x, ComparisonOp::GreaterEqual, k);
  eval(piecewise(vec![(cdf_val, cond)], int(0)))
}

/// PDF[WeibullDistribution[a, b], x] = Piecewise[{{a*(x/b)^(a-1) / (b * E^((x/b)^a)), x > 0}}, 0]
fn pdf_weibull(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "WeibullDistribution expects 2 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();

  // a * (x/b)^(a-1) / (b * E^((x/b)^a))
  let xb = divide(x.clone(), b.clone());
  let numerator = times(a.clone(), power(xb.clone(), minus(a.clone(), int(1))));
  let denom = times(b, power(e(), power(xb, a)));
  let pdf_val = divide(numerator, denom);

  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(pdf_val, cond)], int(0)))
}

/// CDF[WeibullDistribution[a, b], x] = Piecewise[{{1 - E^(-(x/b)^a), x > 0}}, 0]
fn cdf_weibull(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "WeibullDistribution expects 2 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();

  let xb = divide(x.clone(), b);
  let cdf_val = minus(
    int(1),
    power(
      e(),
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(power(xb, a)),
      },
    ),
  );

  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(cdf_val, cond)], int(0)))
}

/// PDF[DiscreteUniformDistribution[{imin, imax}], x] = Piecewise[{{1/(imax-imin+1), imin <= x <= imax}}, 0]
fn pdf_discrete_uniform(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "DiscreteUniformDistribution expects 1 argument".into(),
    ));
  }
  let (imin, imax) = match &dargs[0] {
    Expr::List(bounds) if bounds.len() == 2 => {
      (bounds[0].clone(), bounds[1].clone())
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "DiscreteUniformDistribution expects a list {imin, imax}".into(),
      ));
    }
  };
  let n = eval(plus(minus(imax.clone(), imin.clone()), int(1)))?;
  let pdf_val = divide(int(1), n);
  let cond = comparison3(
    imin,
    ComparisonOp::LessEqual,
    x.clone(),
    ComparisonOp::LessEqual,
    imax,
  );
  eval(piecewise(vec![(pdf_val, cond)], int(0)))
}

/// CDF[DiscreteUniformDistribution[{imin, imax}], x]
fn cdf_discrete_uniform(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "DiscreteUniformDistribution expects 1 argument".into(),
    ));
  }
  let (imin, imax) = match &dargs[0] {
    Expr::List(bounds) if bounds.len() == 2 => {
      (bounds[0].clone(), bounds[1].clone())
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "DiscreteUniformDistribution expects a list {imin, imax}".into(),
      ));
    }
  };
  let n = eval(plus(minus(imax.clone(), imin.clone()), int(1)))?;
  let floor_x = Expr::FunctionCall {
    name: "Floor".to_string(),
    args: vec![x.clone()].into(),
  };
  let cdf_val = divide(plus(minus(floor_x, imin.clone()), int(1)), n);
  let cond_low = comparison(x.clone(), ComparisonOp::Less, imin.clone());
  let cond_mid = comparison3(
    imin,
    ComparisonOp::LessEqual,
    x.clone(),
    ComparisonOp::Less,
    imax.clone(),
  );
  let cond_high = comparison(x, ComparisonOp::GreaterEqual, imax);
  eval(piecewise(
    vec![(int(0), cond_low), (cdf_val, cond_mid), (int(1), cond_high)],
    int(0),
  ))
}

/// PDF[LaplaceDistribution[mu, b], x] = E^(-Abs[x-mu]/b) / (2*b)
fn pdf_laplace(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LaplaceDistribution expects 2 arguments".into(),
    ));
  }
  let mu = dargs[0].clone();
  let b = dargs[1].clone();
  let abs_diff = Expr::FunctionCall {
    name: "Abs".to_string(),
    args: vec![minus(x, mu)].into(),
  };
  let pdf_val = divide(
    power(
      e(),
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(divide(abs_diff, b.clone())),
      },
    ),
    times(int(2), b),
  );
  eval(pdf_val)
}

/// CDF[LaplaceDistribution[mu, b], x] = Piecewise[{{E^((x-mu)/b)/2, x < mu}}, 1 - E^(-(x-mu)/b)/2]
fn cdf_laplace(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LaplaceDistribution expects 2 arguments".into(),
    ));
  }
  let mu = dargs[0].clone();
  let b = dargs[1].clone();
  let diff = minus(x.clone(), mu.clone());
  let low_val = divide(power(e(), divide(diff.clone(), b.clone())), int(2));
  let high_val = minus(
    int(1),
    divide(
      power(
        e(),
        Expr::UnaryOp {
          op: crate::syntax::UnaryOperator::Minus,
          operand: Box::new(divide(diff, b)),
        },
      ),
      int(2),
    ),
  );
  let cond = comparison(x, ComparisonOp::Less, mu);
  eval(piecewise(vec![(low_val, cond)], high_val))
}

/// PDF[RayleighDistribution[sigma], x] = (x/sigma^2) * E^(-x^2/(2*sigma^2)), x > 0
fn pdf_rayleigh(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "RayleighDistribution expects 1 argument".into(),
    ));
  }
  let sigma = dargs[0].clone();
  let s2 = power(sigma, int(2));
  let pdf_val = times(
    divide(x.clone(), s2.clone()),
    power(
      e(),
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(divide(power(x.clone(), int(2)), times(int(2), s2))),
      },
    ),
  );
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(pdf_val, cond)], int(0)))
}

/// CDF[RayleighDistribution[sigma], x] = 1 - E^(-x^2/(2*sigma^2)), x > 0
fn cdf_rayleigh(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "RayleighDistribution expects 1 argument".into(),
    ));
  }
  let sigma = dargs[0].clone();
  let s2 = power(sigma, int(2));
  let cdf_val = minus(
    int(1),
    power(
      e(),
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(divide(power(x.clone(), int(2)), times(int(2), s2))),
      },
    ),
  );
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(cdf_val, cond)], int(0)))
}

/// PDF[MultinomialDistribution[n, {p1, ..., pm}], {x1, ..., xm}]
/// = n! / (x1! * ... * xm!) * p1^x1 * ... * pm^xm when sum(xi) == n and all xi >= 0
/// = 0 otherwise
/// Expressed via products of Binomial coefficients:
/// Binomial[x1+x2, x2] * Binomial[x1+x2+x3, x3] * ... * p1^x1 * ... * pm^xm
fn pdf_multinomial(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "MultinomialDistribution expects 2 arguments".into(),
    ));
  }
  let n = dargs[0].clone();
  let probs = match &dargs[1] {
    Expr::List(items) => items.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "MultinomialDistribution: second argument must be a list".into(),
      ));
    }
  };
  let xs = match &x {
    Expr::List(items) => items.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "PDF[MultinomialDistribution[...], x]: x must be a list".into(),
      ));
    }
  };
  let m = probs.len();
  if xs.len() != m {
    return Err(InterpreterError::EvaluationError(
      "PDF[MultinomialDistribution[...]]: length of x must match length of probabilities".into(),
    ));
  }

  // Build the multinomial coefficient as product of binomials:
  // Binomial[x1+x2, x2] * Binomial[x1+x2+x3, x3] * ...
  // which equals (x1+...+xm)! / (x1! * x2! * ... * xm!)
  let mut coeff: Expr = int(1);
  let mut partial_sum = xs[0].clone();
  for j in 1..m {
    partial_sum = plus(partial_sum, xs[j].clone());
    let binom = Expr::FunctionCall {
      name: "Binomial".to_string(),
      args: vec![partial_sum.clone(), xs[j].clone()].into(),
    };
    coeff = times(coeff, binom);
  }

  // Build product: p1^x1 * p2^x2 * ... * pm^xm
  let mut prob_product = power(probs[0].clone(), xs[0].clone());
  for j in 1..m {
    prob_product = times(prob_product, power(probs[j].clone(), xs[j].clone()));
  }

  let pdf_val = times(coeff, prob_product);

  // Conditions: sum(xi) == n and all xi >= 0
  let sum_xs = {
    let mut s = xs[0].clone();
    for j in 1..m {
      s = plus(s, xs[j].clone());
    }
    s
  };
  let sum_eq_n = comparison(sum_xs, ComparisonOp::Equal, n);

  let mut conditions = vec![sum_eq_n];
  for xi in &xs {
    conditions.push(comparison(xi.clone(), ComparisonOp::GreaterEqual, int(0)));
  }

  // Combine conditions with And
  let combined_cond = if conditions.len() == 1 {
    conditions.remove(0)
  } else {
    Expr::FunctionCall {
      name: "And".to_string(),
      args: conditions.into(),
    }
  };

  eval(piecewise(vec![(pdf_val, combined_cond)], int(0)))
}

/// Returns (Mean list, Variance list) for MultinomialDistribution
/// Mean_i = n * p_i, Variance_i = n * p_i * (1 - p_i)
pub fn multinomial_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "MultinomialDistribution expects 2 arguments".into(),
    ));
  }
  let n = dargs[0].clone();
  let probs = match &dargs[1] {
    Expr::List(items) => items.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "MultinomialDistribution: second argument must be a list".into(),
      ));
    }
  };

  let mut means = Vec::new();
  let mut variances = Vec::new();
  for p in &probs {
    let mean_i = times(n.clone(), p.clone());
    let var_i = times(times(n.clone(), p.clone()), minus(int(1), p.clone()));
    means.push(eval(mean_i)?);
    variances.push(eval(var_i)?);
  }

  Ok((Expr::List(means.into()), Expr::List(variances.into())))
}

/// Mean and Variance for MultivariatePoissonDistribution[theta0, {theta1, ..., thetan}]
/// Mean = {theta0 + theta1, ..., theta0 + thetan}
/// Variance = {theta0 + theta1, ..., theta0 + thetan} (same as mean)
pub fn multivariate_poisson_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "MultivariatePoissonDistribution expects 2 arguments".into(),
    ));
  }
  let theta0 = dargs[0].clone();
  let thetas = match &dargs[1] {
    Expr::List(items) => items.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "MultivariatePoissonDistribution: second argument must be a list"
          .into(),
      ));
    }
  };

  let mut components = Vec::new();
  for theta_i in &thetas {
    let sum = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![theta0.clone(), theta_i.clone()].into(),
    };
    components.push(eval(sum)?);
  }

  // Variance equals Mean for Poisson marginals
  Ok((
    Expr::List(components.clone().into()),
    Expr::List(components.into()),
  ))
}

/// PDF[NegativeBinomialDistribution[n, p], k]
/// = (1-p)^k * p^n * Binomial[k+n-1, n-1] for k >= 0
fn pdf_negative_binomial(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "NegativeBinomialDistribution expects 2 arguments".into(),
    ));
  }
  let n = dargs[0].clone();
  let p = dargs[1].clone();
  // (1-p)^k * p^n * Binomial[-1+k+n, -1+n]
  let one_minus_p = minus(int(1), p.clone());
  let n_minus_1 = plus(int(-1), n.clone());
  let k_plus_n_minus_1 = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![int(-1), x.clone(), n.clone()].into(),
  };
  let binom = Expr::FunctionCall {
    name: "Binomial".to_string(),
    args: vec![k_plus_n_minus_1, n_minus_1].into(),
  };
  let pdf_val = times(times(power(one_minus_p, x.clone()), power(p, n)), binom);
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));
  eval(piecewise(vec![(pdf_val, cond)], int(0)))
}

/// PDF[HalfNormalDistribution[t], x] = Piecewise[{{(2*t)/(E^((t^2*x^2)/Pi)*Pi), x > 0}}, 0]
fn pdf_half_normal(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "HalfNormalDistribution expects 1 argument".into(),
    ));
  }
  let t = dargs[0].clone();
  // (2*t) / (E^((t^2 * x^2) / Pi) * Pi)
  let numerator = times(int(2), t.clone());
  let exponent =
    divide(times(power(t, int(2)), power(x.clone(), int(2))), pi());
  let denominator = times(power(e(), exponent), pi());
  let pdf_val = divide(numerator, denominator);
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(pdf_val, cond)], int(0)))
}

/// CDF[HalfNormalDistribution[t], x] = Piecewise[{{Erf[(t*x)/Sqrt[Pi]], x > 0}}, 0]
fn cdf_half_normal(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "HalfNormalDistribution expects 1 argument".into(),
    ));
  }
  let t = dargs[0].clone();
  let erf_arg = divide(times(t, x.clone()), sqrt(pi()));
  let erf_val = Expr::FunctionCall {
    name: "Erf".to_string(),
    args: vec![erf_arg].into(),
  };
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(erf_val, cond)], int(0)))
}

/// PDF[ChiDistribution[n], x] = Piecewise[{{2^(1-n/2) * x^(n-1) / (E^(x^2/2) * Gamma[n/2]), x > 0}}, 0]
fn pdf_chi(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ChiDistribution expects 1 argument".into(),
    ));
  }
  let n = dargs[0].clone();
  // 2^(1 - n/2)
  let pow2 = power(int(2), minus(int(1), divide(n.clone(), int(2))));
  // x^(n-1)
  let x_pow = power(x.clone(), minus(n.clone(), int(1)));
  // E^(x^2/2)
  let exp_part = power(e(), divide(power(x.clone(), int(2)), int(2)));
  // Gamma[n/2]
  let gamma_part = Expr::FunctionCall {
    name: "Gamma".to_string(),
    args: vec![divide(n, int(2))].into(),
  };
  let pdf_val = eval(divide(times(pow2, x_pow), times(exp_part, gamma_part)))?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(pdf_val, cond)], int(0)))
}

/// CDF[ChiDistribution[n], x] = Piecewise[{{GammaRegularized[n/2, 0, x^2/2], x > 0}}, 0]
fn cdf_chi(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ChiDistribution expects 1 argument".into(),
    ));
  }
  let n = dargs[0].clone();
  let gamma_reg = Expr::FunctionCall {
    name: "GammaRegularized".to_string(),
    args: vec![
      divide(n, int(2)),
      int(0),
      divide(power(x.clone(), int(2)), int(2)),
    ]
    .into(),
  };
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(gamma_reg, cond)], int(0)))
}

/// PDF[StableDistribution[alpha, beta, mu, sigma], x]
/// General case returns unevaluated. Special cases:
/// - alpha=1, beta=0: Cauchy(mu, sigma)
/// - alpha=2: Normal(mu, sigma*Sqrt[2])
fn pdf_stable(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let (alpha, beta, mu, sigma) = match dargs.len() {
    2 => (dargs[0].clone(), dargs[1].clone(), int(0), int(1)),
    4 => (
      dargs[0].clone(),
      dargs[1].clone(),
      dargs[2].clone(),
      dargs[3].clone(),
    ),
    // 5-param canonical form: StableDistribution[1, alpha, beta, mu, sigma]
    5 => (
      dargs[1].clone(),
      dargs[2].clone(),
      dargs[3].clone(),
      dargs[4].clone(),
    ),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PDF".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "StableDistribution".to_string(),
            args: dargs.to_vec().into(),
          },
          x,
        ]
        .into(),
      });
    }
  };

  // Evaluate parameters to check for special cases
  let alpha_eval = evaluate_expr_to_expr(&alpha)?;
  let beta_eval = evaluate_expr_to_expr(&beta)?;

  // alpha=1, beta=0: Cauchy distribution
  // Use expanded form: sigma / (Pi * (sigma^2 + (x - mu)^2))
  // to match Wolfram's canonical simplification
  if matches!(&alpha_eval, Expr::Integer(1))
    && matches!(&beta_eval, Expr::Integer(0))
  {
    let diff = minus(x, mu);
    let numer = sigma.clone();
    let denom = times(pi(), plus(power(sigma, int(2)), power(diff, int(2))));
    return eval(divide(numer, denom));
  }

  // alpha=2: Normal(mu, sigma*Sqrt[2])
  if matches!(&alpha_eval, Expr::Integer(2)) {
    let s = times(sigma, sqrt(int(2)));
    let cauchy_args = vec![mu, s];
    return pdf_normal(&cauchy_args, x);
  }

  // General case: return unevaluated
  Ok(Expr::FunctionCall {
    name: "PDF".to_string(),
    args: vec![
      Expr::FunctionCall {
        name: "StableDistribution".to_string(),
        args: dargs.to_vec().into(),
      },
      x,
    ]
    .into(),
  })
}

/// CDF[StableDistribution[alpha, beta, mu, sigma], x]
fn cdf_stable(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let (alpha, beta, mu, sigma) = match dargs.len() {
    2 => (dargs[0].clone(), dargs[1].clone(), int(0), int(1)),
    4 => (
      dargs[0].clone(),
      dargs[1].clone(),
      dargs[2].clone(),
      dargs[3].clone(),
    ),
    // 5-param canonical form: StableDistribution[1, alpha, beta, mu, sigma]
    5 => (
      dargs[1].clone(),
      dargs[2].clone(),
      dargs[3].clone(),
      dargs[4].clone(),
    ),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "CDF".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "StableDistribution".to_string(),
            args: dargs.to_vec().into(),
          },
          x,
        ]
        .into(),
      });
    }
  };

  let alpha_eval = evaluate_expr_to_expr(&alpha)?;
  let beta_eval = evaluate_expr_to_expr(&beta)?;

  // alpha=1, beta=0: Cauchy CDF = 1/2 + ArcTan[(x-mu)/sigma]/Pi
  if matches!(&alpha_eval, Expr::Integer(1))
    && matches!(&beta_eval, Expr::Integer(0))
  {
    let cauchy_args = vec![mu, sigma];
    return cdf_cauchy(&cauchy_args, x);
  }

  // alpha=2: Normal CDF
  if matches!(&alpha_eval, Expr::Integer(2)) {
    let s = times(sigma, sqrt(int(2)));
    let normal_args = vec![mu, s];
    return cdf_normal(&normal_args, x);
  }

  // General case: return unevaluated
  Ok(Expr::FunctionCall {
    name: "CDF".to_string(),
    args: vec![
      Expr::FunctionCall {
        name: "StableDistribution".to_string(),
        args: dargs.to_vec().into(),
      },
      x,
    ]
    .into(),
  })
}

/// PDF[ArcSinDistribution[{a, b}], x]
/// = Piecewise[{{1/(Pi*Sqrt[(x - a)*(b - x)]), a < x < b}}, 0]
fn pdf_arcsin(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcSinDistribution expects 1 argument (a list {a, b})".into(),
    ));
  }
  let (a, b) = match &dargs[0] {
    Expr::List(bounds) if bounds.len() == 2 => {
      (bounds[0].clone(), bounds[1].clone())
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ArcSinDistribution expects a list {a, b}".into(),
      ));
    }
  };

  // density = 1 / (Pi * Sqrt[(x - a) * (b - x)])
  let density = divide(
    int(1),
    times(
      pi(),
      sqrt(times(
        minus(x.clone(), a.clone()),
        minus(b.clone(), x.clone()),
      )),
    ),
  );

  // condition: a < x < b
  let cond = Expr::Comparison {
    operands: vec![a, x, b],
    operators: vec![
      crate::syntax::ComparisonOp::Less,
      crate::syntax::ComparisonOp::Less,
    ],
  };

  eval(piecewise(vec![(density, cond)], int(0)))
}

/// PDF[PascalDistribution[n, p], k]
/// = (1-p)^(k-n) * p^n * Binomial[k-1, n-1] for k >= n
fn pdf_pascal(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "PascalDistribution expects 2 arguments".into(),
    ));
  }
  let n = dargs[0].clone();
  let p = dargs[1].clone();

  // Binomial[k-1, n-1]
  let binom = Expr::FunctionCall {
    name: "Binomial".to_string(),
    args: vec![minus(x.clone(), int(1)), minus(n.clone(), int(1))].into(),
  };
  // p^n
  let p_n = power(p.clone(), n.clone());
  // (1-p)^(k-n)
  let one_minus_p_k_n = power(minus(int(1), p), minus(x.clone(), n.clone()));

  let density = times(times(binom, p_n), one_minus_p_k_n);

  // k >= n
  let cond = Expr::FunctionCall {
    name: "GreaterEqual".to_string(),
    args: vec![x, n].into(),
  };

  eval(piecewise(vec![(density, cond)], int(0)))
}

/// PDF[DagumDistribution[p, a, b], x]
/// = (a*p/x) * ((x/b)^(a*p)) / (1 + (x/b)^a)^(p+1) for x > 0
fn pdf_dagum(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "DagumDistribution expects 3 arguments".into(),
    ));
  }
  let p = dargs[0].clone();
  let a = dargs[1].clone();
  let b = dargs[2].clone();

  // a*p * x^(a*p - 1) * (1 + (x/b)^a)^(-1-p) / b^(a*p)
  let ap = times(a.clone(), p.clone());
  let x_to_ap_minus_1 = power(x.clone(), minus(ap.clone(), int(1)));
  let x_over_b_to_a = power(divide(x.clone(), b.clone()), a.clone());
  let bracket = power(plus(int(1), x_over_b_to_a), minus(int(-1), p));
  let b_to_ap = power(b, ap.clone());
  let density = divide(times(times(ap, x_to_ap_minus_1), bracket), b_to_ap);

  // x > 0
  let cond = Expr::FunctionCall {
    name: "Greater".to_string(),
    args: vec![x, int(0)].into(),
  };

  eval(piecewise(vec![(density, cond)], int(0)))
}

/// PDF[HyperbolicDistribution[a, b, d, m], x]
/// = Sqrt[a^2 - b^2] * E^(b*(x-m) - a*Sqrt[d^2 + (x-m)^2]) / (2*a*d*BesselK[1, Sqrt[a^2-b^2]*d])
fn pdf_hyperbolic(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 4 {
    return Err(InterpreterError::EvaluationError(
      "HyperbolicDistribution expects 4 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();
  let d = dargs[2].clone();
  let m = dargs[3].clone();

  let besselk = |n: Expr, z: Expr| Expr::FunctionCall {
    name: "BesselK".to_string(),
    args: vec![n, z].into(),
  };

  let x_minus_m = minus(x, m);
  let a2_minus_b2 = minus(power(a.clone(), int(2)), power(b.clone(), int(2)));
  let sqrt_a2_minus_b2 = sqrt(a2_minus_b2);

  // numerator: Sqrt[a^2 - b^2] * E^(b*(x-m) - a*Sqrt[d^2 + (x-m)^2])
  let exponent = minus(
    times(b, x_minus_m.clone()),
    times(
      a.clone(),
      sqrt(plus(power(d.clone(), int(2)), power(x_minus_m, int(2)))),
    ),
  );
  let numerator = times(sqrt_a2_minus_b2.clone(), power(e(), exponent));

  // denominator: 2*a*d*BesselK[1, Sqrt[a^2-b^2]*d]
  let denominator = times(
    times(int(2), times(a, d.clone())),
    besselk(int(1), times(sqrt_a2_minus_b2, d)),
  );

  eval(divide(numerator, denominator))
}

/// PDF[NoncentralFRatioDistribution[n, m, l], x]
/// = m^(m/2)*n^(n/2)*x^((n-2)/2)*(m+n*x)^(-(m+n)/2)
///   *Hypergeometric1F1[(m+n)/2, n/2, l*n*x/(2*(m+n*x))]
///   / (E^(l/2)*Beta[n/2, m/2])
/// for x > 0, else 0
fn pdf_noncentral_f(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "NoncentralFRatioDistribution expects 3 arguments".into(),
    ));
  }
  let n = dargs[0].clone();
  let m = dargs[1].clone();
  let l = dargs[2].clone();

  let half = |e: Expr| divide(e, int(2));

  let beta = |a: Expr, b: Expr| Expr::FunctionCall {
    name: "Beta".to_string(),
    args: vec![a, b].into(),
  };

  let hyp1f1 = |a: Expr, b: Expr, z: Expr| Expr::FunctionCall {
    name: "Hypergeometric1F1".to_string(),
    args: vec![a, b, z].into(),
  };

  // numerator pieces
  let m_half = power(m.clone(), half(m.clone()));
  let n_half = power(n.clone(), half(n.clone()));
  let x_pow = power(x.clone(), half(plus(int(-2), n.clone())));
  let bracket = power(
    plus(m.clone(), times(n.clone(), x.clone())),
    half(plus(times(int(-1), m.clone()), times(int(-1), n.clone()))),
  );
  let hyp = hyp1f1(
    half(plus(m.clone(), n.clone())),
    half(n.clone()),
    divide(
      times(times(l.clone(), n), x.clone()),
      times(int(2), plus(m.clone(), times(dargs[0].clone(), x.clone()))),
    ),
  );

  let numerator =
    times(times(times(m_half, n_half), times(x_pow, bracket)), hyp);

  // denominator: E^(l/2) * Beta[n/2, m/2]
  let denominator =
    times(power(e(), half(l)), beta(half(dargs[0].clone()), half(m)));

  let density = divide(numerator, denominator);

  let cond = comparison(x, ComparisonOp::Greater, int(0));

  eval(piecewise(vec![(density, cond)], int(0)))
}

/// PDF[JohnsonDistribution["type", gamma, delta, mu, sigma], x]
fn pdf_johnson(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 5 {
    return Err(InterpreterError::EvaluationError(
      "JohnsonDistribution expects 5 arguments".into(),
    ));
  }
  let type_str = match &dargs[0] {
    Expr::String(s) => s.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PDF".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "JohnsonDistribution".to_string(),
            args: dargs.to_vec().into(),
          },
          x,
        ]
        .into(),
      });
    }
  };
  let gamma = dargs[1].clone();
  let delta = dargs[2].clone();
  let mu = dargs[3].clone();
  let sigma = dargs[4].clone();

  // (-mu + x): Wolfram canonical ordering
  let neg_mu_plus_x = plus(
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(mu.clone()),
    },
    x.clone(),
  );

  match type_str.as_str() {
    "SN" => {
      // PDF = delta / (E^(z^2/2) * Sqrt[2*Pi] * sigma)
      // where z = gamma + delta*(-mu + x)/sigma
      let z = plus(
        gamma,
        divide(times(delta.clone(), neg_mu_plus_x), sigma.clone()),
      );
      let density = divide(
        delta,
        times(
          times(
            power(e(), divide(power(z, int(2)), int(2))),
            sqrt(times(int(2), pi())),
          ),
          sigma,
        ),
      );
      eval(density)
    }
    "SL" => {
      // PDF = delta / (E^(z^2/2) * Sqrt[2*Pi] * (-mu + x))
      // where z = gamma + delta*Log[(-mu + x)/sigma], for x > mu
      let log_t = Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![divide(neg_mu_plus_x.clone(), sigma)].into(),
      };
      let z = plus(gamma, times(delta.clone(), log_t));
      let density = divide(
        delta,
        times(
          times(
            power(e(), divide(power(z, int(2)), int(2))),
            sqrt(times(int(2), pi())),
          ),
          neg_mu_plus_x,
        ),
      );
      let cond = comparison(x, ComparisonOp::Greater, mu);
      eval(piecewise(vec![(density, cond)], int(0)))
    }
    "SU" => {
      // PDF = delta / (E^(z^2/2) * Sqrt[2*Pi] * Sqrt[sigma^2 + (-mu + x)^2])
      // where z = gamma + delta*ArcSinh[(-mu + x)/sigma]
      let arcsinh_t = Expr::FunctionCall {
        name: "ArcSinh".to_string(),
        args: vec![divide(neg_mu_plus_x.clone(), sigma.clone())].into(),
      };
      let z = plus(gamma, times(delta.clone(), arcsinh_t));
      let density = divide(
        delta,
        times(
          times(
            power(e(), divide(power(z, int(2)), int(2))),
            sqrt(times(int(2), pi())),
          ),
          sqrt(plus(power(sigma, int(2)), power(neg_mu_plus_x, int(2)))),
        ),
      );
      eval(density)
    }
    "SB" => {
      // PDF = (delta*sigma) / (E^(z^2/2) * Sqrt[2*Pi] * (mu+sigma-x) * (-mu+x))
      // where z = gamma + delta*Log[(-mu+x)/(mu+sigma-x)], for mu < x < mu+sigma
      let mu_plus_sigma_minus_x = plus(
        plus(mu.clone(), sigma.clone()),
        Expr::UnaryOp {
          op: crate::syntax::UnaryOperator::Minus,
          operand: Box::new(x.clone()),
        },
      );
      let log_arg =
        divide(neg_mu_plus_x.clone(), mu_plus_sigma_minus_x.clone());
      let log_t = Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![log_arg].into(),
      };
      let z = plus(gamma, times(delta.clone(), log_t));
      let density = divide(
        times(delta, sigma.clone()),
        times(
          times(
            power(e(), divide(power(z, int(2)), int(2))),
            sqrt(times(int(2), pi())),
          ),
          times(mu_plus_sigma_minus_x, neg_mu_plus_x),
        ),
      );
      let cond = comparison3(
        mu.clone(),
        ComparisonOp::Less,
        x.clone(),
        ComparisonOp::Less,
        plus(mu, sigma),
      );
      eval(piecewise(vec![(density, cond)], int(0)))
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "JohnsonDistribution: unknown type {type_str}"
    ))),
  }
}

/// CDF[JohnsonDistribution["type", gamma, delta, mu, sigma], x]
fn cdf_johnson(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 5 {
    return Err(InterpreterError::EvaluationError(
      "JohnsonDistribution expects 5 arguments".into(),
    ));
  }
  let type_str = match &dargs[0] {
    Expr::String(s) => s.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "CDF".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "JohnsonDistribution".to_string(),
            args: dargs.to_vec().into(),
          },
          x,
        ]
        .into(),
      });
    }
  };
  let gamma = dargs[1].clone();
  let delta = dargs[2].clone();
  let mu = dargs[3].clone();
  let sigma = dargs[4].clone();

  // t = (-mu + x) / sigma (canonical ordering)
  let neg_mu_plus_x = plus(
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(mu.clone()),
    },
    x.clone(),
  );
  let t = divide(neg_mu_plus_x.clone(), sigma.clone());

  // CDF = Erfc[(-gamma - delta*h(t)) / Sqrt[2]] / 2
  // where h depends on type
  let h_of_t = match type_str.as_str() {
    "SN" => t.clone(),
    "SL" => Expr::FunctionCall {
      name: "Log".to_string(),
      args: vec![t.clone()].into(),
    },
    "SU" => Expr::FunctionCall {
      name: "ArcSinh".to_string(),
      args: vec![t.clone()].into(),
    },
    "SB" => {
      let mu_plus_sigma_minus_x = plus(
        plus(mu.clone(), sigma.clone()),
        Expr::UnaryOp {
          op: crate::syntax::UnaryOperator::Minus,
          operand: Box::new(x.clone()),
        },
      );
      Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![divide(neg_mu_plus_x, mu_plus_sigma_minus_x)].into(),
      }
    }
    _ => {
      return Err(InterpreterError::EvaluationError(format!(
        "JohnsonDistribution: unknown type {type_str}"
      )));
    }
  };

  // Distribute negative sign: (-gamma - delta*h) / Sqrt[2]
  let neg_gamma = Expr::UnaryOp {
    op: crate::syntax::UnaryOperator::Minus,
    operand: Box::new(gamma.clone()),
  };
  let neg_delta_h = Expr::UnaryOp {
    op: crate::syntax::UnaryOperator::Minus,
    operand: Box::new(times(delta.clone(), h_of_t.clone())),
  };
  let erfc_arg = divide(plus(neg_gamma, neg_delta_h), sqrt(int(2)));
  let cdf_val = divide(
    Expr::FunctionCall {
      name: "Erfc".to_string(),
      args: vec![erfc_arg].into(),
    },
    int(2),
  );

  // Also build (1 + Erf[(gamma + delta*h) / Sqrt[2]]) / 2 form (used by Wolfram for some types)
  let erf_arg = divide(plus(gamma, times(delta, h_of_t)), sqrt(int(2)));
  let cdf_erf = divide(
    plus(
      int(1),
      Expr::FunctionCall {
        name: "Erf".to_string(),
        args: vec![erf_arg].into(),
      },
    ),
    int(2),
  );

  match type_str.as_str() {
    "SN" => eval(cdf_val),
    "SU" => eval(cdf_erf), // Wolfram uses Erf form for SU
    "SL" => {
      // Wolfram splits into two regions based on mu+sigma threshold
      let cond_lower = Expr::Comparison {
        operands: vec![mu.clone(), x.clone(), plus(mu.clone(), sigma.clone())],
        operators: vec![ComparisonOp::Less, ComparisonOp::LessEqual],
      };
      let cond_upper = comparison(x, ComparisonOp::Greater, plus(mu, sigma));
      eval(piecewise(
        vec![(cdf_val, cond_lower), (cdf_erf, cond_upper)],
        int(0),
      ))
    }
    "SB" => {
      // Wolfram splits SB CDF into multiple regions
      let half_sigma = divide(sigma.clone(), int(2));
      let cond1 = comparison3(
        mu.clone(),
        ComparisonOp::Less,
        x.clone(),
        ComparisonOp::Less,
        plus(mu.clone(), half_sigma.clone()),
      );
      let cond2 = Expr::Comparison {
        operands: vec![
          plus(mu.clone(), half_sigma),
          x.clone(),
          plus(mu.clone(), sigma.clone()),
        ],
        operators: vec![ComparisonOp::LessEqual, ComparisonOp::Less],
      };
      let cond3 = comparison(x, ComparisonOp::GreaterEqual, plus(mu, sigma));
      eval(piecewise(
        vec![(cdf_val, cond1), (cdf_erf, cond2), (int(1), cond3)],
        int(0),
      ))
    }
    _ => eval(cdf_val),
  }
}
