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
      // Wolframscript accepts non-negative dimensions; `n == 0` yields an
      // empty list, `{3, 0}` yields three empty inner lists, etc.
      let dims = match &args[1] {
        Expr::Integer(n) if *n >= 0 => vec![*n as usize],
        Expr::List(items) => {
          let mut dims = Vec::new();
          for item in items {
            match item {
              Expr::Integer(n) if *n >= 0 => dims.push(*n as usize),
              _ => {
                return Err(InterpreterError::EvaluationError(
                  "RandomInteger: dimension specification must contain non-negative integers".into(),
                ));
              }
            }
          }
          dims
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "RandomInteger: second argument must be a non-negative integer or list of non-negative integers".into(),
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
          Expr::List(results.into())
        } else {
          let inner_dims = &dims[1..];
          let results: Vec<Expr> = (0..n)
            .map(|_| make_random_int_array(inner_dims, min, max))
            .collect();
          Expr::List(results.into())
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
      // Wolframscript accepts non-negative dimensions; `n == 0` yields an
      // empty list, matching RandomInteger.
      let dims = match &args[1] {
        Expr::Integer(n) if *n >= 0 => vec![*n as usize],
        Expr::List(items) => {
          let mut dims = Vec::new();
          for item in items {
            match item {
              Expr::Integer(n) if *n >= 0 => dims.push(*n as usize),
              _ => {
                return Err(InterpreterError::EvaluationError(
                  "RandomReal: dimension specification must contain non-negative integers".into(),
                ));
              }
            }
          }
          dims
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "RandomReal: second argument must be a non-negative integer or list of non-negative integers".into(),
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
          Expr::List(results.into())
        } else {
          let inner_dims = &dims[1..];
          let results: Vec<Expr> = (0..n)
            .map(|_| make_random_array(inner_dims, min, max))
            .collect();
          Expr::List(results.into())
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
      Expr::FunctionCall { name, args }
        if name == "Complex" && args.len() == 2 =>
      {
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
          InterpreterError::EvaluationError("RandomComplex: invalid min".into())
        })?;
        let (re2, im2) = complex_parts(&items[1]).ok_or_else(|| {
          InterpreterError::EvaluationError("RandomComplex: invalid max".into())
        })?;
        ((re1, re2), (im1, im2))
      }
      _ => {
        let (re, im) = complex_parts(&args[0]).ok_or_else(|| {
          InterpreterError::EvaluationError("RandomComplex: invalid max".into())
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
      // Non-negative dimensions match wolframscript; `n == 0` yields `{}`.
      let dims = match &args[1] {
        Expr::Integer(n) if *n >= 0 => vec![*n as usize],
        Expr::List(items) => {
          let mut dims = Vec::new();
          for item in items {
            match item {
              Expr::Integer(n) if *n >= 0 => dims.push(*n as usize),
              _ => {
                return Err(InterpreterError::EvaluationError(
                  "RandomComplex: dimension specification must contain non-negative integers".into(),
                ));
              }
            }
          }
          dims
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "RandomComplex: second argument must be a non-negative integer or list of non-negative integers".into(),
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

/// RandomColor[] returns a single `RGBColor[r, g, b]` whose three channels
/// are independent uniform draws from [0, 1). `RandomColor[n]` returns a
/// list of `n` such colours.
pub fn random_color_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use rand::Rng;

  fn one_color() -> Expr {
    let (r, g, b) = crate::with_rng(|rng| {
      (
        rng.gen_range(0.0..1.0),
        rng.gen_range(0.0..1.0),
        rng.gen_range(0.0..1.0),
      )
    });
    Expr::FunctionCall {
      name: "RGBColor".to_string(),
      args: vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)].into(),
    }
  }

  match args.len() {
    0 => Ok(one_color()),
    1 => match &args[0] {
      Expr::Integer(n) if *n >= 0 => {
        let count = *n as usize;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
          out.push(one_color());
        }
        Ok(Expr::List(out.into()))
      }
      _ => Ok(Expr::FunctionCall {
        name: "RandomColor".to_string(),
        args: args.to_vec().into(),
      }),
    },
    _ => Ok(Expr::FunctionCall {
      name: "RandomColor".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// `RandomDate[]` returns a single `DateObject` at a random instant in the
/// current calendar year (local time). `RandomDate[n]` returns a list of `n`
/// such DateObjects.
#[cfg(not(target_arch = "wasm32"))]
pub fn random_date_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use chrono::{Datelike, Local, TimeZone, Timelike};
  use rand::Rng;

  fn one_date() -> Expr {
    let now = Local::now();
    let year = now.year();
    let year_start = Local
      .with_ymd_and_hms(year, 1, 1, 0, 0, 0)
      .single()
      .unwrap_or(now);
    let next_year_start = Local
      .with_ymd_and_hms(year + 1, 1, 1, 0, 0, 0)
      .single()
      .unwrap_or(now);
    let span = (next_year_start - year_start).num_milliseconds().max(1) as f64;
    let offset_ms = crate::with_rng(|rng| rng.gen_range(0.0..span));
    let total_micros = (offset_ms * 1000.0) as i64;
    let chosen = year_start + chrono::Duration::microseconds(total_micros);
    let seconds = chosen.second() as f64 + (chosen.nanosecond() as f64) / 1e9;
    let tz_offset_hours = chosen.offset().local_minus_utc() as f64 / 3600.0;
    Expr::FunctionCall {
      name: "DateObject".to_string(),
      args: vec![
        Expr::List(
          vec![
            Expr::Integer(chosen.year() as i128),
            Expr::Integer(chosen.month() as i128),
            Expr::Integer(chosen.day() as i128),
            Expr::Integer(chosen.hour() as i128),
            Expr::Integer(chosen.minute() as i128),
            Expr::Real(seconds),
          ]
          .into(),
        ),
        Expr::String("Instant".to_string()),
        Expr::String("Gregorian".to_string()),
        Expr::Real(tz_offset_hours),
      ]
      .into(),
    }
  }

  match args.len() {
    0 => Ok(one_date()),
    1 => match &args[0] {
      Expr::Integer(n) if *n >= 0 => {
        let count = *n as usize;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
          out.push(one_date());
        }
        Ok(Expr::List(out.into()))
      }
      _ => Ok(Expr::FunctionCall {
        name: "RandomDate".to_string(),
        args: args.to_vec().into(),
      }),
    },
    _ => Ok(Expr::FunctionCall {
      name: "RandomDate".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

#[cfg(target_arch = "wasm32")]
pub fn random_date_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  Ok(Expr::FunctionCall {
    name: "RandomDate".to_string(),
    args: args.to_vec().into(),
  })
}

/// Random[] or Random[Real] / Random[Integer] / Random[Complex],
/// optionally followed by a range argument. This is the legacy wrapper
/// around RandomReal / RandomInteger / RandomComplex.
pub fn random_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Random[] → RandomReal[]
  if args.is_empty() {
    return random_real_ast(&[]);
  }

  // The first argument is the type; the rest (optional) is the range.
  let kind = match &args[0] {
    Expr::Identifier(s) => s.as_str(),
    _ => {
      // Random[foo] where foo isn't a type symbol — treat foo as a max arg
      // to RandomReal for compatibility.
      return random_real_ast(args);
    }
  };
  let rest: Vec<Expr> = args.iter().skip(1).cloned().collect();

  match kind {
    "Real" => random_real_ast(&rest),
    "Integer" => random_integer_ast(&rest),
    "Complex" => random_complex_ast(&rest),
    _ => Ok(Expr::FunctionCall {
      name: "Random".to_string(),
      args: args.to_vec().into(),
    }),
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

  // Weighted form: RandomChoice[weights -> values, …]. Build a CDF and
  // pick by drawing a uniform u in [0, total) and binary-searching the
  // first prefix sum >= u. Falls back to unevaluated when the rule's
  // shape isn't valid.
  let (items_owned, weighted_cdf): (Vec<Expr>, Option<Vec<f64>>) =
    match &args[0] {
      Expr::Rule {
        pattern,
        replacement,
      } => {
        let (Expr::List(weights), Expr::List(values)) =
          (pattern.as_ref(), replacement.as_ref())
        else {
          return Ok(Expr::FunctionCall {
            name: "RandomChoice".to_string(),
            args: args.to_vec().into(),
          });
        };
        if weights.is_empty() || weights.len() != values.len() {
          return Err(InterpreterError::EvaluationError(
            "RandomChoice: weight and value lists must have matching length"
              .into(),
          ));
        }
        let mut cdf = Vec::with_capacity(weights.len());
        let mut acc = 0.0f64;
        for w in weights.iter() {
          let v = expr_to_num(w).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "RandomChoice: weights must be numeric".into(),
            )
          })?;
          if v < 0.0 {
            return Err(InterpreterError::EvaluationError(
              "RandomChoice: weights must be non-negative".into(),
            ));
          }
          acc += v;
          cdf.push(acc);
        }
        if acc <= 0.0 {
          return Err(InterpreterError::EvaluationError(
            "RandomChoice: weights must sum to a positive value".into(),
          ));
        }
        (values.iter().cloned().collect(), Some(cdf))
      }
      _ => (Vec::new(), None),
    };
  let items: &[Expr] = if weighted_cdf.is_some() {
    &items_owned
  } else {
    match &args[0] {
      Expr::List(items) if !items.is_empty() => items.as_ref(),
      Expr::List(_) => {
        return Err(InterpreterError::EvaluationError(
          "RandomChoice: list cannot be empty".into(),
        ));
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "RandomChoice".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  };
  // Helper: pick one index given the (optional) weight CDF.
  let pick_index = |rng: &mut dyn rand::RngCore| -> usize {
    if let Some(cdf) = weighted_cdf.as_ref() {
      let total = *cdf.last().unwrap();
      let u = rng.gen_range(0.0..total);
      cdf.iter().position(|&c| u < c).unwrap_or(cdf.len() - 1)
    } else {
      rng.gen_range(0..items.len())
    }
  };

  if args.len() == 1 {
    let idx = crate::with_rng(pick_index);
    Ok(items[idx].clone())
  } else {
    // Accept either a non-negative integer (flat list) or a list of
    // positive integers (nested array of that shape). Matches Wolfram.
    let dims: Vec<usize> = match &args[1] {
      Expr::Integer(n) if *n >= 0 => vec![*n as usize],
      Expr::List(ds) if !ds.is_empty() => {
        let mut out = Vec::with_capacity(ds.len());
        for d in ds.iter() {
          match d {
            Expr::Integer(k) if *k > 0 => out.push(*k as usize),
            _ => {
              return Err(InterpreterError::EvaluationError(
                "RandomChoice: dimension entries must be positive integers"
                  .into(),
              ));
            }
          }
        }
        out
      }
      _ => {
        return Err(InterpreterError::EvaluationError(
          "RandomChoice: second argument must be a non-negative integer or list of dimensions".into(),
        ));
      }
    };
    fn build(items: &[Expr], dims: &[usize], cdf: Option<&[f64]>) -> Expr {
      use rand::Rng;
      let n = dims[0];
      if dims.len() == 1 {
        let result: Vec<Expr> = crate::with_rng(|rng| {
          (0..n)
            .map(|_| {
              let idx = if let Some(cdf) = cdf {
                let total = *cdf.last().unwrap();
                let u = rng.gen_range(0.0..total);
                cdf.iter().position(|&c| u < c).unwrap_or(cdf.len() - 1)
              } else {
                rng.gen_range(0..items.len())
              };
              items[idx].clone()
            })
            .collect()
        });
        Expr::List(result.into())
      } else {
        let inner = &dims[1..];
        let result: Vec<Expr> =
          (0..n).map(|_| build(items, inner, cdf)).collect();
        Expr::List(result.into())
      }
    }
    Ok(build(items, &dims, weighted_cdf.as_deref()))
  }
}

/// RandomSample[list]
pub fn random_sample_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use rand::Rng;
  use rand::seq::SliceRandom;

  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "RandomSample expects 1 or 2 arguments".into(),
    ));
  }

  // Weighted form: RandomSample[weights -> values, n]. Sample without
  // replacement using the standard "exponential trick" (Efraimidis–Spirakis):
  // assign each item a key u_i^(1/w_i) where u_i ~ Uniform(0, 1), then pick
  // the n items with the largest keys. Equivalent to weighted reservoir
  // sampling without replacement, matching Wolfram's semantics.
  let (items_owned, weights): (Vec<Expr>, Option<Vec<f64>>) = match &args[0] {
    Expr::Rule {
      pattern,
      replacement,
    } => {
      let (Expr::List(weights), Expr::List(values)) =
        (pattern.as_ref(), replacement.as_ref())
      else {
        return Ok(Expr::FunctionCall {
          name: "RandomSample".to_string(),
          args: args.to_vec().into(),
        });
      };
      if weights.is_empty() || weights.len() != values.len() {
        return Err(InterpreterError::EvaluationError(
          "RandomSample: weight and value lists must have matching length"
            .into(),
        ));
      }
      let mut ws = Vec::with_capacity(weights.len());
      for w in weights.iter() {
        let v = expr_to_num(w).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "RandomSample: weights must be numeric".into(),
          )
        })?;
        if v < 0.0 {
          return Err(InterpreterError::EvaluationError(
            "RandomSample: weights must be non-negative".into(),
          ));
        }
        ws.push(v);
      }
      (values.iter().cloned().collect(), Some(ws))
    }
    _ => (Vec::new(), None),
  };

  let items: &[Expr] = if weights.is_some() {
    &items_owned
  } else {
    match &args[0] {
      Expr::List(items) => items.as_ref(),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "RandomSample".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  };

  if args.len() == 1 {
    // Weighted shuffle: full ordering by Efraimidis–Spirakis keys when
    // weights are supplied, otherwise plain uniform shuffle.
    if let Some(ws) = weights.as_ref() {
      let keys: Vec<f64> = crate::with_rng(|rng| {
        ws.iter()
          .map(|&w| {
            let u: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
            if w <= 0.0 {
              f64::NEG_INFINITY
            } else {
              u.ln() / w
            }
          })
          .collect()
      });
      let mut idxs: Vec<usize> = (0..items.len()).collect();
      idxs.sort_by(|&a, &b| {
        keys[b]
          .partial_cmp(&keys[a])
          .unwrap_or(std::cmp::Ordering::Equal)
      });
      let result: Vec<Expr> =
        idxs.into_iter().map(|i| items[i].clone()).collect();
      Ok(Expr::List(result.into()))
    } else {
      let mut shuffled: Vec<Expr> = items.to_vec();
      crate::with_rng(|rng| shuffled.shuffle(rng));
      Ok(Expr::List(shuffled.into()))
    }
  } else {
    let n = match &args[1] {
      Expr::Integer(n) if *n >= 0 => *n as usize,
      _ => {
        // Match wolframscript: emit RandomSample::intnm and return the
        // call unevaluated rather than aborting. Covers the dim-list
        // form (e.g. {2, 3}) which Wolfram doesn't support either.
        let arg_strs: Vec<String> =
          args.iter().map(crate::syntax::expr_to_output).collect();
        crate::emit_message(&format!(
          "RandomSample::intnm: Non-negative machine-sized integer expected at position 2 in RandomSample[{}].",
          arg_strs.join(", "),
        ));
        return Ok(Expr::FunctionCall {
          name: "RandomSample".to_string(),
          args: args.to_vec().into(),
        });
      }
    };
    if n > items.len() {
      return Err(InterpreterError::EvaluationError(format!(
        "RandomSample: cannot sample {} elements from list of length {}",
        n,
        items.len()
      )));
    }
    if let Some(ws) = weights.as_ref() {
      // Pick the n items with the largest Efraimidis–Spirakis keys.
      let keys: Vec<f64> = crate::with_rng(|rng| {
        ws.iter()
          .map(|&w| {
            let u: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
            if w <= 0.0 {
              f64::NEG_INFINITY
            } else {
              u.ln() / w
            }
          })
          .collect()
      });
      let mut idxs: Vec<usize> = (0..items.len()).collect();
      idxs.sort_by(|&a, &b| {
        keys[b]
          .partial_cmp(&keys[a])
          .unwrap_or(std::cmp::Ordering::Equal)
      });
      idxs.truncate(n);
      let sampled: Vec<Expr> =
        idxs.into_iter().map(|i| items[i].clone()).collect();
      Ok(Expr::List(sampled.into()))
    } else {
      let sampled: Vec<Expr> =
        crate::with_rng(|rng| items.choose_multiple(rng, n).cloned().collect());
      Ok(Expr::List(sampled.into()))
    }
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

  // Local helpers for the distributions added below.
  fn sample_poisson(lambda: f64) -> Result<i128, InterpreterError> {
    use rand_distr::{Distribution, Poisson};
    let p = Poisson::new(lambda).map_err(|e| {
      InterpreterError::EvaluationError(format!("PoissonDistribution: {}", e))
    })?;
    Ok(crate::with_rng(|rng| p.sample(rng) as i128))
  }
  fn sample_binormal(rho: f64) -> (f64, f64) {
    use rand_distr::{Distribution, Normal};
    let n01 = Normal::new(0.0, 1.0).unwrap();
    let (z1, z2) = crate::with_rng(|rng| (n01.sample(rng), n01.sample(rng)));
    let y = rho * z1 + (1.0 - rho * rho).max(0.0).sqrt() * z2;
    (z1, y)
  }

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
                Ok(Expr::List(results.into()))
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
          Ok(Expr::List(results.into()))
        }
      }
    }
    Expr::FunctionCall { name, args: dargs }
      if name == "PoissonDistribution" && dargs.len() == 1 =>
    {
      let lambda = expr_to_num(&dargs[0]).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "PoissonDistribution: invalid mean parameter".into(),
        )
      })?;
      if lambda <= 0.0 {
        return Err(InterpreterError::EvaluationError(
          "PoissonDistribution: mean must be positive".into(),
        ));
      }
      match n {
        None => Ok(Expr::Integer(sample_poisson(lambda)?)),
        Some(count) => {
          let mut out = Vec::with_capacity(count);
          for _ in 0..count {
            out.push(Expr::Integer(sample_poisson(lambda)?));
          }
          Ok(Expr::List(out.into()))
        }
      }
    }
    Expr::FunctionCall { name, args: dargs }
      if name == "BinormalDistribution" =>
    {
      // BinormalDistribution[rho] uses standard means and unit variances.
      // BinormalDistribution[{mu1, mu2}, {sigma1, sigma2}, rho] is the full form.
      let (mu1, mu2, sigma1, sigma2, rho) = match dargs.len() {
        1 => {
          let r = expr_to_num(&dargs[0]).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "BinormalDistribution: invalid correlation".into(),
            )
          })?;
          (0.0, 0.0, 1.0, 1.0, r)
        }
        3 => {
          let extract_pair =
            |e: &Expr, label: &str| -> Result<(f64, f64), InterpreterError> {
              if let Expr::List(items) = e
                && items.len() == 2
                && let (Some(a), Some(b)) =
                  (expr_to_num(&items[0]), expr_to_num(&items[1]))
              {
                Ok((a, b))
              } else {
                Err(InterpreterError::EvaluationError(format!(
                  "BinormalDistribution: invalid {} (expected a 2-element list)",
                  label
                )))
              }
            };
          let (mu1, mu2) = extract_pair(&dargs[0], "mean vector")?;
          let (sigma1, sigma2) =
            extract_pair(&dargs[1], "standard deviations")?;
          let r = expr_to_num(&dargs[2]).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "BinormalDistribution: invalid correlation".into(),
            )
          })?;
          (mu1, mu2, sigma1, sigma2, r)
        }
        _ => {
          return Ok(Expr::FunctionCall {
            name: "RandomVariate".to_string(),
            args: args.to_vec().into(),
          });
        }
      };
      if !(-1.0..=1.0).contains(&rho) {
        return Err(InterpreterError::EvaluationError(
          "BinormalDistribution: correlation must be in [-1, 1]".into(),
        ));
      }
      let sample_pair = || -> (f64, f64) {
        let (z1, z2) = sample_binormal(rho);
        (mu1 + sigma1 * z1, mu2 + sigma2 * z2)
      };
      let pair_to_list = |p: (f64, f64)| -> Expr {
        Expr::List(vec![Expr::Real(p.0), Expr::Real(p.1)].into())
      };
      match n {
        None => Ok(pair_to_list(sample_pair())),
        Some(count) => {
          let mut out = Vec::with_capacity(count);
          for _ in 0..count {
            out.push(pair_to_list(sample_pair()));
          }
          Ok(Expr::List(out.into()))
        }
      }
    }
    Expr::FunctionCall { name, args: dargs }
      if name == "MultivariatePoissonDistribution" && dargs.len() == 2 =>
    {
      // MultivariatePoissonDistribution[mu0, {mu1, ..., muk}]:
      //   X_0 ~ Poisson(mu0) shared across components;
      //   Y_i ~ Poisson(mu_i) independent;
      //   sample is (X_0 + Y_1, ..., X_0 + Y_k).
      let mu0 = expr_to_num(&dargs[0]).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "MultivariatePoissonDistribution: invalid shared rate".into(),
        )
      })?;
      let marginals_list = if let Expr::List(items) = &dargs[1] {
        items
      } else {
        return Err(InterpreterError::EvaluationError(
          "MultivariatePoissonDistribution: expected a list of marginal rates"
            .into(),
        ));
      };
      let mut marginal_rates: Vec<f64> =
        Vec::with_capacity(marginals_list.len());
      for item in marginals_list.iter() {
        let mi = expr_to_num(item).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "MultivariatePoissonDistribution: invalid marginal rate".into(),
          )
        })?;
        if mi < mu0 {
          return Err(InterpreterError::EvaluationError(format!(
            "MultivariatePoissonDistribution: each marginal must be >= mu0 ({} < {})",
            mi, mu0
          )));
        }
        marginal_rates.push(mi);
      }
      let sample_vec = || -> Result<Expr, InterpreterError> {
        let x0 = sample_poisson(mu0)?;
        let mut comps = Vec::with_capacity(marginal_rates.len());
        for &mi in &marginal_rates {
          let yi = sample_poisson(mi - mu0)?;
          comps.push(Expr::Integer(x0 + yi));
        }
        Ok(Expr::List(comps.into()))
      };
      match n {
        None => sample_vec(),
        Some(count) => {
          let mut out = Vec::with_capacity(count);
          for _ in 0..count {
            out.push(sample_vec()?);
          }
          Ok(Expr::List(out.into()))
        }
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "RandomVariate".to_string(),
      args: args.to_vec().into(),
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
        args: args.to_vec().into(),
      });
    }
  };

  // The second arg may be a positive integer (length of a flat list)
  // or a list of dimensions {n1, n2, ...} for a nested array. Track
  // both as a `dims` vector — when it has length 1 we keep the
  // original flat-list output to match wolframscript.
  let dims: Vec<usize> = if args.len() == 2 {
    match &args[1] {
      Expr::Integer(n) if *n > 0 => vec![*n as usize],
      Expr::List(items) if !items.is_empty() => {
        let mut ds = Vec::with_capacity(items.len());
        for it in items.iter() {
          match it {
            Expr::Integer(n) if *n > 0 => ds.push(*n as usize),
            _ => {
              return Err(InterpreterError::EvaluationError(
                "RandomPrime: dimension entries must be positive integers"
                  .into(),
              ));
            }
          }
        }
        ds
      }
      _ => {
        return Err(InterpreterError::EvaluationError(
          "RandomPrime: second argument must be a positive integer or list of dimensions".into(),
        ));
      }
    }
  } else {
    Vec::new()
  };
  // Build a flat-then-nested list of `total` primes, then reshape into
  // the requested dims (skipping reshape when dims is empty/scalar or a
  // single-dim list).
  fn reshape_into_dims(flat: Vec<Expr>, dims: &[usize]) -> Expr {
    if dims.len() <= 1 {
      return Expr::List(flat.into());
    }
    let chunk = flat.len() / dims[0];
    let mut groups: Vec<Expr> = Vec::with_capacity(dims[0]);
    for i in 0..dims[0] {
      let part: Vec<Expr> = flat[i * chunk..(i + 1) * chunk].to_vec();
      groups.push(reshape_into_dims(part, &dims[1..]));
    }
    Expr::List(groups.into())
  }
  let total: usize = if dims.is_empty() {
    1
  } else {
    dims.iter().product()
  };
  let count = total;

  let range_size = max - min + 1;

  // For small ranges, enumerate primes; for large ranges, use rejection sampling
  if range_size <= 100_000 {
    let primes = collect_primes_in_range(min, max);
    if primes.is_empty() {
      return Err(InterpreterError::EvaluationError(
        "There are no primes in the specified interval.".into(),
      ));
    }
    if dims.is_empty() {
      let idx = crate::with_rng(|rng| rng.gen_range(0..primes.len()));
      Ok(Expr::Integer(primes[idx]))
    } else {
      let flat: Vec<Expr> = crate::with_rng(|rng| {
        (0..count)
          .map(|_| Expr::Integer(primes[rng.gen_range(0..primes.len())]))
          .collect()
      });
      Ok(reshape_into_dims(flat, &dims))
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
    if dims.is_empty() {
      Ok(results.into_iter().next().unwrap())
    } else {
      Ok(reshape_into_dims(results, &dims))
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

/// RandomGraph[{n, m}] — undirected graph with n vertices and m random edges.
/// RandomGraph[{n, m}, k] — list of k such graphs.
pub fn random_graph_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use rand::seq::SliceRandom;
  use rand::Rng;

  let unevaluated = || Expr::FunctionCall {
    name: "RandomGraph".to_string(),
    args: args.to_vec().into(),
  };

  // Split positional args from trailing Rule options.
  let mut positional: Vec<&Expr> = Vec::new();
  let mut options: Vec<Expr> = Vec::new();
  for a in args {
    if matches!(a, Expr::Rule { .. }) {
      options.push(a.clone());
    } else {
      positional.push(a);
    }
  }

  if positional.is_empty() {
    return Ok(unevaluated());
  }

  // RandomGraph[BernoulliGraphDistribution[n, p]] — Erdős–Rényi G(n, p):
  // each potential edge is independently included with probability p.
  if let Expr::FunctionCall { name, args: bargs } = positional[0]
    && name == "BernoulliGraphDistribution"
    && bargs.len() == 2
  {
    let n = match crate::functions::math_ast::expr_to_i128(&bargs[0]) {
      Some(n) if n >= 0 => n as usize,
      _ => return Ok(unevaluated()),
    };
    let p = match crate::functions::math_ast::try_eval_to_f64(&bargs[1]) {
      Some(p) if (0.0..=1.0).contains(&p) => p,
      _ => return Ok(unevaluated()),
    };
    let mut edges: Vec<(i128, i128)> = Vec::new();
    crate::with_rng(|rng| {
      for i in 1..=n {
        for j in (i + 1)..=n {
          if rng.gen_bool(p) {
            edges.push((i as i128, j as i128));
          }
        }
      }
    });
    let vertex_list: Vec<Expr> =
      (1..=n).map(|v| Expr::Integer(v as i128)).collect();
    let edge_list: Vec<Expr> = edges
      .into_iter()
      .map(|(a, b)| Expr::FunctionCall {
        name: "UndirectedEdge".to_string(),
        args: vec![Expr::Integer(a), Expr::Integer(b)].into(),
      })
      .collect();
    let mut graph_args =
      vec![Expr::List(vertex_list.into()), Expr::List(edge_list.into())];
    if !options.is_empty() {
      graph_args.push(Expr::List(options.into()));
    }
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: graph_args.into(),
    });
  }

  let (n, m) = match positional[0] {
    Expr::List(items) if items.len() == 2 => {
      match (
        crate::functions::math_ast::expr_to_i128(&items[0]),
        crate::functions::math_ast::expr_to_i128(&items[1]),
      ) {
        (Some(n), Some(m)) if n >= 0 && m >= 0 => (n as usize, m as usize),
        _ => return Ok(unevaluated()),
      }
    }
    _ => return Ok(unevaluated()),
  };

  // Max number of edges in K_n.
  let max_edges = n.saturating_mul(n.saturating_sub(1)) / 2;
  if m > max_edges {
    return Ok(unevaluated());
  }

  let k = match positional.len() {
    1 => None,
    2 => match crate::functions::math_ast::expr_to_i128(positional[1]) {
      Some(k) if k >= 0 => Some(k as usize),
      _ => return Ok(unevaluated()),
    },
    _ => return Ok(unevaluated()),
  };

  // Pre-build all possible edges {1≤i<j≤n}. For practical n this fits.
  let mut all_edges: Vec<(i128, i128)> = Vec::with_capacity(max_edges);
  for i in 1..=n {
    for j in (i + 1)..=n {
      all_edges.push((i as i128, j as i128));
    }
  }

  let opts_list = if options.is_empty() {
    None
  } else {
    Some(Expr::List(options.into()))
  };
  let make_graph = |edges: Vec<(i128, i128)>| -> Expr {
    let vertex_list: Vec<Expr> =
      (1..=n).map(|v| Expr::Integer(v as i128)).collect();
    let edge_list: Vec<Expr> = edges
      .into_iter()
      .map(|(a, b)| Expr::FunctionCall {
        name: "UndirectedEdge".to_string(),
        args: vec![Expr::Integer(a), Expr::Integer(b)].into(),
      })
      .collect();
    let mut graph_args =
      vec![Expr::List(vertex_list.into()), Expr::List(edge_list.into())];
    if let Some(opts) = &opts_list {
      graph_args.push(opts.clone());
    }
    Expr::FunctionCall {
      name: "Graph".to_string(),
      args: graph_args.into(),
    }
  };

  let one_graph = |all_edges: &[(i128, i128)]| -> Expr {
    let mut shuffled = all_edges.to_vec();
    crate::with_rng(|rng| shuffled.shuffle(rng));
    shuffled.truncate(m);
    shuffled.sort();
    make_graph(shuffled)
  };

  match k {
    None => Ok(one_graph(&all_edges)),
    Some(k) => {
      let graphs: Vec<Expr> = (0..k).map(|_| one_graph(&all_edges)).collect();
      Ok(Expr::List(graphs.into()))
    }
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
