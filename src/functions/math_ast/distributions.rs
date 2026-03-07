#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
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

  Ok(piecewise(vec![(density, cond)], int(0)))
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

  Ok(piecewise(vec![(density, cond)], int(0)))
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

  Ok(piecewise(vec![(density, cond)], int(0)))
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

  Ok(piecewise(vec![(one_minus_p, cond0), (p, cond1)], int(0)))
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

  Ok(piecewise(
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

  Ok(piecewise(vec![(value, cond)], int(0)))
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

  Ok(piecewise(
    vec![(int(0), cond_neg), (one_minus_p, cond_middle)],
    int(1),
  ))
}
