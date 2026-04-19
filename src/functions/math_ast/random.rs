#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;

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

/// RandomComplex[] or RandomComplex[max] or RandomComplex[{min, max}]
/// or RandomComplex[range, dims]. Real and imaginary parts are drawn
/// uniformly from the specified rectangle.
pub fn random_complex_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use crate::syntax::BinaryOperator;
  use rand::Rng;

  fn complex_parts(expr: &Expr) -> Option<(f64, f64)> {
    // Extract (re, im) from an Expr that evaluates to a number.
    match expr {
      Expr::Integer(n) => Some((*n as f64, 0.0)),
      Expr::Real(r) => Some((*r, 0.0)),
      Expr::Identifier(s) if s == "I" => Some((0.0, 1.0)),
      Expr::FunctionCall { name, args } if name == "Complex" && args.len() == 2 => {
        let re = match &args[0] {
          Expr::Integer(n) => *n as f64,
          Expr::Real(r) => *r,
          _ => return None,
        };
        let im = match &args[1] {
          Expr::Integer(n) => *n as f64,
          Expr::Real(r) => *r,
          _ => return None,
        };
        Some((re, im))
      }
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left,
        right,
      } => {
        let (re1, im1) = complex_parts(left)?;
        let (re2, im2) = complex_parts(right)?;
        Some((re1 + re2, im1 + im2))
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        let (re1, im1) = complex_parts(left)?;
        let (re2, im2) = complex_parts(right)?;
        Some((re1 * re2 - im1 * im2, re1 * im2 + im1 * re2))
      }
      Expr::FunctionCall { name, args } if name == "Plus" => {
        let mut re = 0.0;
        let mut im = 0.0;
        for a in args {
          let (r, i) = complex_parts(a)?;
          re += r;
          im += i;
        }
        Some((re, im))
      }
      Expr::FunctionCall { name, args } if name == "Times" => {
        let mut re = 1.0;
        let mut im = 0.0;
        for a in args {
          let (r, i) = complex_parts(a)?;
          let nr = re * r - im * i;
          let ni = re * i + im * r;
          re = nr;
          im = ni;
        }
        Some((re, im))
      }
      _ => None,
    }
  }

  fn make_complex(re: f64, im: f64) -> Expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(Expr::Real(re)),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Real(im)),
        right: Box::new(Expr::Identifier("I".to_string())),
      }),
    }
  }

  // Parse the range from the first argument (default: 0 to 1+I).
  let ((re_lo, re_hi), (im_lo, im_hi)) = if args.is_empty() {
    ((0.0, 1.0), (0.0, 1.0))
  } else {
    match &args[0] {
      Expr::List(items) if items.len() == 2 => {
        let (re1, im1) = complex_parts(&items[0]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "RandomComplex: invalid min".into(),
          )
        })?;
        let (re2, im2) = complex_parts(&items[1]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "RandomComplex: invalid max".into(),
          )
        })?;
        ((re1, re2), (im1, im2))
      }
      _ => {
        let (re, im) = complex_parts(&args[0]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "RandomComplex: invalid max".into(),
          )
        })?;
        ((0.0, re), (0.0, im))
      }
    }
  };

  fn draw_one(
    rng: &mut dyn rand::RngCore,
    re_lo: f64,
    re_hi: f64,
    im_lo: f64,
    im_hi: f64,
  ) -> Expr {
    let re = if re_lo == re_hi {
      re_lo
    } else {
      rng.gen_range(re_lo..re_hi)
    };
    let im = if im_lo == im_hi {
      im_lo
    } else {
      rng.gen_range(im_lo..im_hi)
    };
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left: Box::new(Expr::Real(re)),
      right: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(Expr::Real(im)),
        right: Box::new(Expr::Identifier("I".to_string())),
      }),
    }
  }

  let _ = make_complex; // silence unused warning if code paths change

  match args.len() {
    0 | 1 => Ok(crate::with_rng(|rng| {
      draw_one(rng, re_lo, re_hi, im_lo, im_hi)
    })),
    2 => {
      let dims = match &args[1] {
        Expr::Integer(n) if *n > 0 => vec![*n as usize],
        Expr::List(items) => {
          let mut dims = Vec::new();
          for item in items {
            match item {
              Expr::Integer(n) if *n > 0 => dims.push(*n as usize),
              _ => {
                return Err(InterpreterError::EvaluationError(
                  "RandomComplex: dimension specification must contain positive integers".into(),
                ));
              }
            }
          }
          dims
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "RandomComplex: second argument must be a positive integer or list of positive integers".into(),
          ));
        }
      };

      fn build(
        dims: &[usize],
        re_lo: f64,
        re_hi: f64,
        im_lo: f64,
        im_hi: f64,
      ) -> Expr {
        let n = dims[0];
        if dims.len() == 1 {
          Expr::List(crate::with_rng(|rng| {
            (0..n)
              .map(|_| draw_one(rng, re_lo, re_hi, im_lo, im_hi))
              .collect()
          }))
        } else {
          Expr::List(
            (0..n)
              .map(|_| build(&dims[1..], re_lo, re_hi, im_lo, im_hi))
              .collect(),
          )
        }
      }

      Ok(build(&dims, re_lo, re_hi, im_lo, im_hi))
    }
    _ => Err(InterpreterError::EvaluationError(
      "RandomComplex expects 0, 1, or 2 arguments".into(),
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

/// RandomPrime[imax] or RandomPrime[{imin, imax}] or RandomPrime[range, n]
pub fn random_prime_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use rand::Rng;

  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "RandomPrime expects 1 or 2 arguments".into(),
    ));
  }

  // Parse the range from first argument
  let (min, max) = match &args[0] {
    Expr::Integer(imax) => {
      if *imax < 2 {
        return Err(InterpreterError::EvaluationError(
          "There are no primes in the specified interval.".into(),
        ));
      }
      (2i128, *imax)
    }
    Expr::List(items) if items.len() == 2 => {
      if let (Expr::Integer(imin), Expr::Integer(imax)) = (&items[0], &items[1])
      {
        (*imin, *imax)
      } else {
        return Err(InterpreterError::EvaluationError(
          "RandomPrime: range bounds must be integers".into(),
        ));
      }
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "RandomPrime".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let count = if args.len() == 2 {
    match &args[1] {
      Expr::Integer(n) if *n > 0 => *n as usize,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "RandomPrime: second argument must be a positive integer".into(),
        ));
      }
    }
  } else {
    1
  };

  let range_size = max - min + 1;

  // For small ranges, enumerate primes; for large ranges, use rejection sampling
  if range_size <= 100_000 {
    let primes = collect_primes_in_range(min, max);
    if primes.is_empty() {
      return Err(InterpreterError::EvaluationError(
        "There are no primes in the specified interval.".into(),
      ));
    }
    if count == 1 {
      let idx = crate::with_rng(|rng| rng.gen_range(0..primes.len()));
      Ok(Expr::Integer(primes[idx]))
    } else {
      let result: Vec<Expr> = crate::with_rng(|rng| {
        (0..count)
          .map(|_| Expr::Integer(primes[rng.gen_range(0..primes.len())]))
          .collect()
      });
      Ok(Expr::List(result))
    }
  } else {
    // Rejection sampling for large ranges
    let start = if min < 2 { 2i128 } else { min };
    if start > max {
      return Err(InterpreterError::EvaluationError(
        "There are no primes in the specified interval.".into(),
      ));
    }
    let mut results = Vec::with_capacity(count);
    for _ in 0..count {
      let prime = crate::with_rng(|rng| {
        loop {
          let candidate = rng.gen_range(start..=max);
          if is_prime_i128(candidate) {
            return candidate;
          }
        }
      });
      results.push(Expr::Integer(prime));
    }
    if count == 1 {
      Ok(results.into_iter().next().unwrap())
    } else {
      Ok(Expr::List(results))
    }
  }
}

/// Collect all primes in [min, max] (for small ranges)
fn collect_primes_in_range(min: i128, max: i128) -> Vec<i128> {
  let start = if min < 2 { 2 } else { min };
  let mut primes = Vec::new();
  for n in start..=max {
    if crate::is_prime(n as usize) {
      primes.push(n);
    }
  }
  primes
}

/// Primality test for i128 values, using Miller-Rabin for large numbers
fn is_prime_i128(n: i128) -> bool {
  if n < 2 {
    return false;
  }
  if n <= usize::MAX as i128 {
    return crate::is_prime(n as usize);
  }
  // For values beyond usize, use BigInt Miller-Rabin
  let big = num_bigint::BigInt::from(n);
  super::number_theory::is_prime_bigint(&big)
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
      Expr::String(s) => {
        // Wolfram accepts string seeds; hash the string deterministically
        // so the same string always produces the same sequence.
        crate::seed_rng(hash_string_to_u64(s));
        Ok(Expr::Identifier("Null".to_string()))
      }
      _ => Err(InterpreterError::EvaluationError(
        "SeedRandom: seed must be an integer or string".into(),
      )),
    },
    _ => Err(InterpreterError::EvaluationError(
      "SeedRandom expects 0 or 1 arguments".into(),
    )),
  }
}

/// Deterministically derive a 64-bit seed from a string. The same string
/// always maps to the same seed, so `SeedRandom["foo"]` is reproducible
/// across runs and across platforms.
fn hash_string_to_u64(s: &str) -> u64 {
  // FNV-1a 64-bit — stable, simple, not dependent on Rust's std hasher
  // (whose output may differ between versions/platforms).
  let mut hash: u64 = 0xcbf29ce484222325;
  for byte in s.as_bytes() {
    hash ^= *byte as u64;
    hash = hash.wrapping_mul(0x100000001b3);
  }
  hash
}
