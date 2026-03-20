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
  Expr::FunctionCall {
    name: "Sqrt".to_string(),
    args: vec![a],
  }
}

fn factorial(a: Expr) -> Expr {
  Expr::FunctionCall {
    name: "Factorial".to_string(),
    args: vec![a],
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
      .map(|(val, cond)| Expr::List(vec![val, cond]))
      .collect(),
  );
  Expr::FunctionCall {
    name: "Piecewise".to_string(),
    args: vec![cases, default],
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
        args: args.to_vec(),
      });
    }
  };

  if args.len() == 1 {
    // PDF[dist] - return unevaluated for now
    return Ok(Expr::FunctionCall {
      name: "PDF".to_string(),
      args: args.to_vec(),
    });
  }

  let x = args[1].clone();

  match dist_name {
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
    _ => Ok(Expr::FunctionCall {
      name: "PDF".to_string(),
      args: args.to_vec(),
    }),
  }
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
        args: args.to_vec(),
      });
    }
  };

  if args.len() == 1 {
    return Ok(Expr::FunctionCall {
      name: "CDF".to_string(),
      args: args.to_vec(),
    });
  }

  let x = args[1].clone();

  match dist_name {
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
    _ => Ok(Expr::FunctionCall {
      name: "CDF".to_string(),
      args: args.to_vec(),
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
    args: vec![erfc_arg],
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
      args: vec![x.clone()],
    },
    int(1),
  );
  let value = Expr::FunctionCall {
    name: "GammaRegularized".to_string(),
    args: vec![floor_k_plus_1, mu],
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
      args: vec![x.clone()],
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
    args: vec![divide(minus(x, a), b)],
  };
  eval(plus(
    Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![int(1), int(2)],
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
      args: vec![alpha],
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
    args: vec![alpha, int(0), divide(x.clone(), beta)],
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
      args: vec![a],
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
    args: vec![a, divide(b, x.clone())],
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

  // (x^(-1))^(1 + n/2) = x^(-(1 + n/2))
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
    args: vec![divide(n, int(2))],
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
    args: vec![divide(n, int(2)), divide(int(1), times(int(2), x.clone()))],
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
  eval(divide(power(e(), exp_arg), b))
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
      args: vec![erfc1_arg],
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
      args: vec![erfc2_arg],
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
/// event can be: x > a, x < a, x >= a, x <= a, x == k, a < x < b, etc.
pub fn probability_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Probability expects 2 arguments".into(),
    ));
  }

  let event = &args[0];
  let dist_spec = &args[1];

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
          args: args.to_vec(),
        });
      }
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Probability".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Determine if distribution is discrete
  let is_discrete = matches!(
    dist,
    Expr::FunctionCall { name, .. }
    if matches!(name.as_str(), "PoissonDistribution" | "BernoulliDistribution" | "BinomialDistribution" | "GeometricDistribution" | "NegativeBinomialDistribution" | "DiscreteUniformDistribution")
  );

  // Parse the event condition and compute probability
  let result = probability_from_event(event, var_name, dist, is_discrete)?;
  // Apply Together to normalize fractions (e.g. 1 - E^(-2) → (-1 + E^2)/E^2)
  eval(Expr::FunctionCall {
    name: "Together".to_string(),
    args: vec![result],
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
        args: vec![Expr::Identifier(var.to_string()), dist.clone()],
      },
    ],
  })
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

  // Parse Distributed[var, dist]
  let (var_name, dist) = match dist_spec {
    Expr::FunctionCall { name, args: dargs }
      if name == "Distributed" && dargs.len() == 2 =>
    {
      if let Expr::Identifier(v) = &dargs[0] {
        (v.clone(), &dargs[1])
      } else {
        return Ok(Expr::FunctionCall {
          name: "Expectation".to_string(),
          args: args.to_vec(),
        });
      }
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Expectation".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Get distribution name and parameters
  let (dist_name, dargs) = match dist {
    Expr::FunctionCall { name, args: da } => (name.as_str(), da.as_slice()),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Expectation".to_string(),
        args: args.to_vec(),
      });
    }
  };

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
          args: vec![plus(int(1), divide(int(1), a.clone()))],
        },
      );
      // Var = b^2 * (Gamma[1 + 2/a] - Gamma[1 + 1/a]^2)
      let var = times(
        power(b, int(2)),
        minus(
          Expr::FunctionCall {
            name: "Gamma".to_string(),
            args: vec![plus(int(1), divide(int(2), a.clone()))],
          },
          power(
            Expr::FunctionCall {
              name: "Gamma".to_string(),
              args: vec![plus(int(1), divide(int(1), a))],
            },
            int(2),
          ),
        ),
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
      let one_minus_p = minus(int(1), p.clone());
      let mean = divide(times(n.clone(), one_minus_p.clone()), p.clone());
      let var = divide(times(n, one_minus_p), power(p, int(2)));
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
            args: coeff_parts,
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
                args: dargs.to_vec(),
              },
            ],
          },
        ],
      });
    }
  };

  // Numerical integration: E[f(x)] = integral f(x) * pdf(x) dx
  let dx = (hi - lo) / n_points as f64;
  let mut sum = 0.0;
  let dist_expr = Expr::FunctionCall {
    name: dist_name.to_string(),
    args: dargs.to_vec(),
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
    args: vec![a, b],
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
    args: vec![x.clone(), a, b],
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
      args: vec![divide(nu, int(2)), divide(int(1), int(2))],
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
    args: vec![x.clone()],
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
    args: vec![x.clone()],
  };
  let arg = Expr::UnaryOp {
    op: crate::syntax::UnaryOperator::Minus,
    operand: Box::new(divide(minus(log_x, mu), times(sqrt(int(2)), sigma))),
  };
  let cdf_val = divide(
    Expr::FunctionCall {
      name: "Erfc".to_string(),
      args: vec![arg],
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
    args: vec![divide(k, int(2))],
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
    args: vec![divide(k, int(2)), int(0), divide(x.clone(), int(2))],
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
    args: vec![x.clone()],
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
    args: vec![minus(x, mu)],
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
      args: vec![partial_sum.clone(), xs[j].clone()],
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
      args: conditions,
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

  Ok((Expr::List(means), Expr::List(variances)))
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
  // (1-p)^k * p^n * Binomial[k+n-1, n-1]
  let one_minus_p = minus(int(1), p.clone());
  let binom = Expr::FunctionCall {
    name: "Binomial".to_string(),
    args: vec![
      plus(x.clone(), minus(n.clone(), int(1))),
      minus(n.clone(), int(1)),
    ],
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
    args: vec![erf_arg],
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
    args: vec![divide(n, int(2))],
  };
  let pdf_val = divide(times(pow2, x_pow), times(exp_part, gamma_part));
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
    ],
  };
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(gamma_reg, cond)], int(0)))
}
