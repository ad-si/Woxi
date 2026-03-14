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
    "GammaDistribution" => pdf_gamma(dargs, x),
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
    "GammaDistribution" => cdf_gamma(dargs, x),
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
    if matches!(name.as_str(), "PoissonDistribution" | "BernoulliDistribution" | "BinomialDistribution" | "GeometricDistribution")
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
  if let Expr::FunctionCall { name, args } = event {
    if name == "And" && args.len() == 2 {
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
  {
    if operands.len() == 2 && operators.len() == 1 {
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
      if !contains_var(right, var) {
        if let Some((a, b)) = extract_linear(left, var) {
          let new_b = plus(b, right.as_ref().clone());
          return Some((a, new_b));
        }
      }
      if !contains_var(left, var) {
        if let Some((a, b)) = extract_linear(right, var) {
          let new_b = plus(left.as_ref().clone(), b);
          return Some((a, new_b));
        }
      }
      None
    }
    Expr::BinaryOp { op, left, right }
      if *op == crate::syntax::BinaryOperator::Minus =>
    {
      if !contains_var(right, var) {
        if let Some((a, b)) = extract_linear(left, var) {
          let new_b = minus(b, right.as_ref().clone());
          return Some((a, new_b));
        }
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
