use crate::InterpreterError;
use crate::syntax::{Expr};
#[allow(unused_imports)]
use super::*;

/// RandomInteger[max] or RandomInteger[{min, max}] - Random integer
pub fn random_integer_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use rand::Rng;

  match args.len() {
    0 => Ok(Expr::Integer(crate::with_rng(|rng| rng.gen_range(0..=1)))),
    1 => match &args[0] {
      Expr::Integer(max) => {
        if *max < 0 {
          Err(InterpreterError::EvaluationError(
            "RandomInteger: max must be non-negative".into(),
          ))
        } else {
          let max = *max;
          Ok(Expr::Integer(crate::with_rng(|rng| rng.gen_range(0..=max))))
        }
      }
      Expr::List(items) if items.len() == 2 => {
        if let (Expr::Integer(min), Expr::Integer(max)) = (&items[0], &items[1])
        {
          if min > max {
            Err(InterpreterError::EvaluationError(
              "RandomInteger: min must be <= max".into(),
            ))
          } else {
            let (min, max) = (*min, *max);
            Ok(Expr::Integer(crate::with_rng(|rng| {
              rng.gen_range(min..=max)
            })))
          }
        } else {
          Err(InterpreterError::EvaluationError(
            "RandomInteger: range must be integers".into(),
          ))
        }
      }
      _ => Err(InterpreterError::EvaluationError(
        "RandomInteger: invalid argument".into(),
      )),
    },
    2 => {
      // RandomInteger[range, n] or RandomInteger[range, {n}] or RandomInteger[range, {n, m, ...}]
      let dims = match &args[1] {
        Expr::Integer(n) if *n > 0 => vec![*n as usize],
        Expr::List(items) => {
          let mut dims = Vec::new();
          for item in items {
            match item {
              Expr::Integer(n) if *n > 0 => dims.push(*n as usize),
              _ => {
                return Err(InterpreterError::EvaluationError(
                  "RandomInteger: dimension specification must contain positive integers".into(),
                ));
              }
            }
          }
          dims
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "RandomInteger: second argument must be a positive integer or list of positive integers".into(),
          ));
        }
      };

      let (min, max) = match &args[0] {
        Expr::Integer(m) => (0i128, *m),
        Expr::List(items) if items.len() == 2 => {
          if let (Expr::Integer(min), Expr::Integer(max)) =
            (&items[0], &items[1])
          {
            (*min, *max)
          } else {
            return Err(InterpreterError::EvaluationError(
              "RandomInteger: range must be integers".into(),
            ));
          }
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "RandomInteger: invalid range".into(),
          ));
        }
      };

      if min > max {
        return Err(InterpreterError::EvaluationError(
          "RandomInteger: min must be <= max".into(),
        ));
      }

      fn make_random_int_array(dims: &[usize], min: i128, max: i128) -> Expr {
        let n = dims[0];
        if dims.len() == 1 {
          let results: Vec<Expr> = crate::with_rng(|rng| {
            (0..n)
              .map(|_| Expr::Integer(rng.gen_range(min..=max)))
              .collect()
          });
          Expr::List(results)
        } else {
          let inner_dims = &dims[1..];
          let results: Vec<Expr> = (0..n)
            .map(|_| make_random_int_array(inner_dims, min, max))
            .collect();
          Expr::List(results)
        }
      }

      Ok(make_random_int_array(&dims, min, max))
    }
    _ => Err(InterpreterError::EvaluationError(
      "RandomInteger expects 0, 1, or 2 arguments".into(),
    )),
  }
}

/// RandomReal[] or RandomReal[max] or RandomReal[{min, max}]
pub fn random_real_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use rand::Rng;

  match args.len() {
    0 => Ok(Expr::Real(crate::with_rng(|rng| rng.gen_range(0.0..1.0)))),
    1 => match &args[0] {
      Expr::Integer(max) => {
        let max = *max as f64;
        Ok(Expr::Real(crate::with_rng(|rng| {
          rng.gen_range(0.0..1.0) * max
        })))
      }
      Expr::Real(max) => {
        let max = *max;
        Ok(Expr::Real(crate::with_rng(|rng| {
          rng.gen_range(0.0..1.0) * max
        })))
      }
      Expr::List(items) if items.len() == 2 => {
        let min = expr_to_num(&items[0]).ok_or_else(|| {
          InterpreterError::EvaluationError("RandomReal: invalid min".into())
        })?;
        let max = expr_to_num(&items[1]).ok_or_else(|| {
          InterpreterError::EvaluationError("RandomReal: invalid max".into())
        })?;
        Ok(Expr::Real(crate::with_rng(|rng| {
          min + rng.gen_range(0.0..1.0) * (max - min)
        })))
      }
      _ => Err(InterpreterError::EvaluationError(
        "RandomReal: invalid argument".into(),
      )),
    },
    2 => {
      // RandomReal[range, n] or RandomReal[range, {n}] or RandomReal[range, {n, m, ...}]
      let dims = match &args[1] {
        Expr::Integer(n) if *n > 0 => vec![*n as usize],
        Expr::List(items) => {
          let mut dims = Vec::new();
          for item in items {
            match item {
              Expr::Integer(n) if *n > 0 => dims.push(*n as usize),
              _ => {
                return Err(InterpreterError::EvaluationError(
                  "RandomReal: dimension specification must contain positive integers".into(),
                ));
              }
            }
          }
          dims
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "RandomReal: second argument must be a positive integer or list of positive integers".into(),
          ));
        }
      };

      let (min, max) = match &args[0] {
        Expr::Integer(m) => (0.0, *m as f64),
        Expr::Real(m) => (0.0, *m),
        Expr::List(items) if items.len() == 2 => {
          let lo = expr_to_num(&items[0]).ok_or_else(|| {
            InterpreterError::EvaluationError("RandomReal: invalid min".into())
          })?;
          let hi = expr_to_num(&items[1]).ok_or_else(|| {
            InterpreterError::EvaluationError("RandomReal: invalid max".into())
          })?;
          (lo, hi)
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "RandomReal: invalid range".into(),
          ));
        }
      };

      fn make_random_array(dims: &[usize], min: f64, max: f64) -> Expr {
        let n = dims[0];
        if dims.len() == 1 {
          let results: Vec<Expr> = crate::with_rng(|rng| {
            (0..n)
              .map(|_| Expr::Real(min + rng.gen_range(0.0..1.0) * (max - min)))
              .collect()
          });
          Expr::List(results)
        } else {
          let inner_dims = &dims[1..];
          let results: Vec<Expr> = (0..n)
            .map(|_| make_random_array(inner_dims, min, max))
            .collect();
          Expr::List(results)
        }
      }

      Ok(make_random_array(&dims, min, max))
    }
    _ => Err(InterpreterError::EvaluationError(
      "RandomReal expects 0, 1, or 2 arguments".into(),
    )),
  }
}

/// RandomChoice[list]
pub fn random_choice_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use rand::Rng;

  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "RandomChoice expects 1 or 2 arguments".into(),
    ));
  }

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    Expr::List(_) => {
      return Err(InterpreterError::EvaluationError(
        "RandomChoice: list cannot be empty".into(),
      ));
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "RandomChoice".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if args.len() == 1 {
    let idx = crate::with_rng(|rng| rng.gen_range(0..items.len()));
    Ok(items[idx].clone())
  } else {
    let n = match &args[1] {
      Expr::Integer(n) if *n >= 0 => *n as usize,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "RandomChoice: second argument must be a non-negative integer".into(),
        ));
      }
    };
    let result: Vec<Expr> = crate::with_rng(|rng| {
      (0..n)
        .map(|_| {
          let idx = rng.gen_range(0..items.len());
          items[idx].clone()
        })
        .collect()
    });
    Ok(Expr::List(result))
  }
}

/// RandomSample[list]
pub fn random_sample_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use rand::seq::SliceRandom;

  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "RandomSample expects 1 or 2 arguments".into(),
    ));
  }

  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "RandomSample".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if args.len() == 1 {
    let mut shuffled = items.clone();
    crate::with_rng(|rng| shuffled.shuffle(rng));
    Ok(Expr::List(shuffled))
  } else {
    let n = match &args[1] {
      Expr::Integer(n) if *n >= 0 => *n as usize,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "RandomSample: second argument must be a non-negative integer".into(),
        ));
      }
    };
    if n > items.len() {
      return Err(InterpreterError::EvaluationError(format!(
        "RandomSample: cannot sample {} elements from list of length {}",
        n,
        items.len()
      )));
    }
    let sampled: Vec<Expr> =
      crate::with_rng(|rng| items.choose_multiple(rng, n).cloned().collect());
    Ok(Expr::List(sampled))
  }
}

/// RandomVariate[dist] or RandomVariate[dist, n]
/// Supports UniformDistribution[{min, max}] and NormalDistribution[mu, sigma]
pub fn random_variate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use rand::Rng;
  use rand_distr::{Distribution, Normal};

  let dist = &args[0];
  let n = if args.len() == 2 {
    match &args[1] {
      Expr::Integer(n) if *n > 0 => Some(*n as usize),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "RandomVariate: second argument must be a positive integer".into(),
        ));
      }
    }
  } else {
    None
  };

  // Determine distribution type and parameters, then sample using with_rng
  match dist {
    Expr::FunctionCall { name, args: dargs }
      if name == "UniformDistribution" =>
    {
      if dargs.len() == 1 {
        if let Expr::List(bounds) = &dargs[0] {
          if bounds.len() == 2 {
            let lo = expr_to_num(&bounds[0]).ok_or_else(|| {
              InterpreterError::EvaluationError(
                "UniformDistribution: invalid min bound".into(),
              )
            })?;
            let hi = expr_to_num(&bounds[1]).ok_or_else(|| {
              InterpreterError::EvaluationError(
                "UniformDistribution: invalid max bound".into(),
              )
            })?;
            match n {
              None => {
                Ok(Expr::Real(crate::with_rng(|rng| rng.gen_range(lo..hi))))
              }
              Some(count) => {
                let results: Vec<Expr> = crate::with_rng(|rng| {
                  (0..count)
                    .map(|_| Expr::Real(rng.gen_range(lo..hi)))
                    .collect()
                });
                Ok(Expr::List(results))
              }
            }
          } else {
            Err(InterpreterError::EvaluationError(
              "UniformDistribution: expected {min, max}".into(),
            ))
          }
        } else {
          Err(InterpreterError::EvaluationError(
            "UniformDistribution: expected a list {min, max}".into(),
          ))
        }
      } else {
        Err(InterpreterError::EvaluationError(
          "UniformDistribution: expected 1 argument".into(),
        ))
      }
    }
    Expr::FunctionCall { name, args: dargs }
      if name == "NormalDistribution" =>
    {
      let (mu, sigma) = match dargs.len() {
        0 => (0.0, 1.0),
        2 => {
          let mu = expr_to_num(&dargs[0]).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "NormalDistribution: invalid mean".into(),
            )
          })?;
          let sigma = expr_to_num(&dargs[1]).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "NormalDistribution: invalid standard deviation".into(),
            )
          })?;
          (mu, sigma)
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "NormalDistribution: expected 0 or 2 arguments".into(),
          ));
        }
      };
      let normal = Normal::new(mu, sigma).map_err(|e| {
        InterpreterError::EvaluationError(format!("NormalDistribution: {}", e))
      })?;
      match n {
        None => Ok(Expr::Real(crate::with_rng(|rng| normal.sample(rng)))),
        Some(count) => {
          let results: Vec<Expr> = crate::with_rng(|rng| {
            (0..count).map(|_| Expr::Real(normal.sample(rng))).collect()
          });
          Ok(Expr::List(results))
        }
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "RandomVariate".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// SeedRandom[n] - Seed the random number generator
/// SeedRandom[] - Reset to non-deterministic RNG
pub fn seed_random_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match args.len() {
    0 => {
      crate::unseed_rng();
      Ok(Expr::Identifier("Null".to_string()))
    }
    1 => match &args[0] {
      Expr::Integer(seed) => {
        crate::seed_rng(*seed as u64);
        Ok(Expr::Identifier("Null".to_string()))
      }
      _ => Err(InterpreterError::EvaluationError(
        "SeedRandom: seed must be an integer".into(),
      )),
    },
    _ => Err(InterpreterError::EvaluationError(
      "SeedRandom expects 0 or 1 arguments".into(),
    )),
  }
}

