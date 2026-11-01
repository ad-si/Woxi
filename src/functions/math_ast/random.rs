#[allow(unused_imports)]
use super::*;

/// Emit the `<Name>::array` message and return the call unevaluated for an
/// invalid dimension specification (a negative integer, a non-integer, or a
/// list containing one) in position 2 of a Random* array generator, matching
/// wolframscript instead of raising a hard evaluation error.
fn random_array_dims_error(name: &str, args: &[Expr]) -> Expr {
  let call = unevaluated(name, args);
  crate::emit_message(&format!(
    "{}::array: The array dimensions {} given in position 2 of {} should be a list of non-negative machine-sized integers giving the dimensions for the result.",
    name,
    expr_to_string(&args[1]),
    expr_to_string(&call)
  ));
  call
}

/// The arguments with a dimension spec in position 2 given as whole-valued
/// reals (`RandomReal[1, {50., 50.}]`, `n = 50.` used as a count) read as
/// the integers they are, which Wolfram accepts; any other spec is left for
/// the caller to validate.
pub(crate) fn integral_dims(args: &[Expr]) -> std::borrow::Cow<'_, [Expr]> {
  fn whole(e: &Expr) -> Option<Expr> {
    match e {
      Expr::Real(f) if f.fract() == 0.0 && *f >= 0.0 && *f < 9.0e15 => {
        Some(Expr::Integer(*f as i128))
      }
      _ => None,
    }
  }
  let Some(dims) = args.get(1) else {
    return std::borrow::Cow::Borrowed(args);
  };
  let normalized = match dims {
    Expr::Real(_) => whole(dims),
    Expr::List(items) if items.iter().any(|i| matches!(i, Expr::Real(_))) => {
      items
        .iter()
        .map(|i| match i {
          Expr::Real(_) => whole(i),
          other => Some(other.clone()),
        })
        .collect::<Option<Vec<_>>>()
        .map(|v| Expr::List(v.into()))
    }
    _ => None,
  };
  match normalized {
    Some(d) => {
      let mut v = args.to_vec();
      v[1] = d;
      std::borrow::Cow::Owned(v)
    }
    None => std::borrow::Cow::Borrowed(args),
  }
}

/// RandomInteger[max] or RandomInteger[{min, max}] - Random integer
pub fn random_integer_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let args = &*integral_dims(args);
  use rand::RngExt;

  // RandomInteger[dist] / RandomInteger[dist, n] samples from a discrete
  // distribution, exactly like RandomVariate (e.g. `RandomInteger[
  // PoissonDistribution[10], 1000]`). Delegate to RandomVariate so every
  // distribution it supports works here too.
  if let Some(Expr::FunctionCall { name, .. }) = args.first()
    && name.ends_with("Distribution")
  {
    return random_variate_ast(args);
  }

  if let Some(result) = random_big_integer(args) {
    return result;
  }

  match args.len() {
    0 => Ok(Expr::Integer(crate::with_rng(|rng| {
      rng.random_range(0..=1)
    }))),
    1 => match &args[0] {
      // `RandomInteger[n]` draws from 0 through n — which for a negative n is
      // the range n through 0, not an error.
      Expr::Integer(max) => {
        let (lo, hi) = if *max < 0 { (*max, 0) } else { (0, *max) };
        Ok(Expr::Integer(crate::with_rng(|rng| {
          rng.random_range(lo..=hi)
        })))
      }
      Expr::List(items) if items.len() == 2 => {
        if let (Expr::Integer(min), Expr::Integer(max)) = (&items[0], &items[1])
        {
          // A reversed range {max, min} is ordered before sampling, matching
          // wolframscript (RandomInteger[{5, 2}] draws from [2, 5]).
          let (lo, hi) = (*min.min(max), *min.max(max));
          Ok(Expr::Integer(crate::with_rng(|rng| {
            rng.random_range(lo..=hi)
          })))
        } else {
          Err(InterpreterError::EvaluationError(
            "RandomInteger: range must be integers".into(),
          ))
        }
      }
      // `RandomInteger[{imax}]` omits `imin`, which defaults to 0 — same as
      // the bare `RandomInteger[imax]` form (wolframscript accepts both).
      Expr::List(items) if items.len() == 1 => {
        if let Expr::Integer(max) = &items[0] {
          let (lo, hi) = if *max < 0 { (*max, 0) } else { (0, *max) };
          Ok(Expr::Integer(crate::with_rng(|rng| {
            rng.random_range(lo..=hi)
          })))
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
              _ => return Ok(random_array_dims_error("RandomInteger", args)),
            }
          }
          dims
        }
        _ => return Ok(random_array_dims_error("RandomInteger", args)),
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
        // `RandomInteger[{imax}, dims]` omits `imin`, which defaults to 0.
        Expr::List(items) if items.len() == 1 => {
          if let Expr::Integer(max) = &items[0] {
            (0i128, *max)
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

      // A reversed range is ordered before sampling (wolframscript parity).
      let (min, max) = (min.min(max), min.max(max));

      fn make_random_int_array(dims: &[usize], min: i128, max: i128) -> Expr {
        let n = dims[0];
        if dims.len() == 1 {
          let results: Vec<Expr> = crate::with_rng(|rng| {
            (0..n)
              .map(|_| Expr::Integer(rng.random_range(min..=max)))
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

/// `RandomInteger` over a range with a bound beyond machine integers
/// (`RandomInteger[10^50]`, `RandomInteger[{-10^40, 10^40}, 3]`). `None`
/// when no bound is a big integer, leaving the machine path in charge.
fn random_big_integer(args: &[Expr]) -> Option<Result<Expr, InterpreterError>> {
  use num_bigint::BigInt;
  let big = |e: &Expr| match e {
    Expr::Integer(n) => Some(BigInt::from(*n)),
    Expr::BigInteger(n) => Some(n.clone()),
    _ => None,
  };
  let spec = args.first()?;
  let bounds: Vec<&Expr> = match spec {
    Expr::List(items) => items.iter().collect(),
    other => vec![other],
  };
  if !bounds.iter().any(|b| matches!(b, Expr::BigInteger(_))) {
    return None;
  }
  let (lo, hi) = match bounds.as_slice() {
    [max] => {
      let max = big(max)?;
      let zero = BigInt::from(0);
      if max < zero { (max, zero) } else { (zero, max) }
    }
    [a, b] => {
      let (a, b) = (big(a)?, big(b)?);
      if a <= b { (a, b) } else { (b, a) }
    }
    _ => return None,
  };
  let span = &hi - &lo + 1u32;
  let bits = span.bits();
  // Rejection sampling of a uniform value below `span`.
  let draw = || -> Expr {
    let value = crate::with_rng(|rng| {
      use rand::RngExt;
      loop {
        let words = bits.div_ceil(32) as usize;
        let mut digits: Vec<u32> =
          (0..words).map(|_| rng.random::<u32>()).collect();
        let extra = (words as u64) * 32 - bits;
        if let Some(top) = digits.last_mut() {
          *top >>= extra;
        }
        let r = BigInt::from(num_bigint::BigUint::new(digits));
        if r < span {
          return r;
        }
      }
    });
    crate::functions::math_ast::bigint_to_expr(&lo + value)
  };
  fn fill(dims: &[usize], draw: &dyn Fn() -> Expr) -> Expr {
    match dims.split_first() {
      None => draw(),
      Some((&n, rest)) => {
        Expr::List((0..n).map(|_| fill(rest, draw)).collect::<Vec<_>>().into())
      }
    }
  }
  let dims: Vec<usize> = match args.get(1) {
    None => vec![],
    Some(Expr::Integer(n)) if *n >= 0 => vec![*n as usize],
    Some(Expr::List(items)) => items
      .iter()
      .map(|i| match i {
        Expr::Integer(n) if *n >= 0 => Some(*n as usize),
        _ => None,
      })
      .collect::<Option<Vec<_>>>()?,
    Some(_) => return None,
  };
  if args.len() > 2 {
    return None;
  }
  Some(Ok(fill(&dims, &draw)))
}

/// RandomReal[] or RandomReal[max] or RandomReal[{min, max}]
pub fn random_real_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let args = &*integral_dims(args);
  use rand::RngExt;

  // `RandomReal` takes options — `WorkingPrecision -> p` above all — after
  // its range and dimensions. (`RandomInteger` does not: wolframscript
  // reports `RandomInteger::argb` for a third argument.) The precision only
  // changes how many digits are asked for, and Woxi samples machine reals,
  // so the option is read and dropped.
  let (args, working_precision) = split_random_options(args);
  let args = args.as_slice();
  if let Some(p) = working_precision.as_ref().and_then(extended_precision) {
    // A uniform draw at a precision beyond machine numbers.
    if let Some(bounds) = match args.first() {
      None => Some((Expr::Integer(0), Expr::Integer(1))),
      Some(Expr::FunctionCall { name, .. })
        if name.ends_with("Distribution") =>
      {
        None
      }
      Some(Expr::List(items)) if items.len() == 2 => {
        Some((items[0].clone(), items[1].clone()))
      }
      Some(max) => Some((Expr::Integer(0), max.clone())),
    } && let Some(dims) = random_dims(args.get(1))
    {
      return fill_with(&dims, &mut || {
        uniform_at_precision(&bounds.0, &bounds.1, p)
      });
    }
  }

  // RandomReal[dist] / RandomReal[dist, n] samples from a continuous
  // distribution, exactly like RandomVariate (e.g. `RandomReal[
  // UniformDistribution[{-2, 2}], 10]`). Delegate to RandomVariate so
  // every distribution it supports works here too — the same delegation
  // `RandomInteger` makes for the discrete ones.
  if let Some(Expr::FunctionCall { name, .. }) = args.first()
    && name.ends_with("Distribution")
  {
    return random_variate_ast(args);
  }

  // A distribution of the caller's own: Wolfram draws from anything that
  // defines `Random`DistributionVector[dist, n, precision]`, which is how a
  // Demonstration adds a distribution of its own without touching the
  // built-in list. `RandomReal[dist]` is one draw — the first row of a
  // vector of length one, so a distribution over points gives a point.
  if let Some(dist @ Expr::FunctionCall { name, .. }) = args.first()
    && !name.ends_with("Distribution")
    && args.len() <= 2
    && let Some(count) = match args.get(1) {
      None => Some(None),
      Some(Expr::Integer(n)) if *n >= 0 => Some(Some(*n)),
      Some(_) => None,
    }
    && let Some(drawn) = distribution_vector_draw(dist, count.unwrap_or(1))
  {
    return Ok(match (count, &drawn) {
      (None, Expr::List(rows)) => {
        rows.first().cloned().unwrap_or(Expr::List(vec![].into()))
      }
      _ => drawn,
    });
  }

  // `{min, max}` or `max`; anything not real valued is reported (unifr)
  // and leaves the call unevaluated, as in wolframscript.
  let range = |spec: &Expr| -> Option<(f64, f64)> {
    let bounds = match spec {
      Expr::List(items) if items.len() == 2 => {
        try_eval_to_f64(&items[0]).zip(try_eval_to_f64(&items[1]))
      }
      other => try_eval_to_f64(other).map(|max| (0.0, max)),
    };
    if bounds.is_none() {
      crate::emit_message(&format!(
        "RandomReal::unifr: The endpoints specified by {} for the endpoints \
         of the uniform distribution range are not real valued.",
        crate::syntax::format_expr(spec, crate::syntax::ExprForm::Output)
      ));
    }
    bounds
  };

  match args.len() {
    0 => Ok(Expr::Real(crate::with_rng(|rng| {
      rng.random_range(0.0..1.0)
    }))),
    1 => {
      let Some((min, max)) = range(&args[0]) else {
        return Ok(unevaluated("RandomReal", args));
      };
      Ok(Expr::Real(crate::with_rng(|rng| {
        min + rng.random_range(0.0..1.0) * (max - min)
      })))
    }
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
              _ => return Ok(random_array_dims_error("RandomReal", args)),
            }
          }
          dims
        }
        _ => return Ok(random_array_dims_error("RandomReal", args)),
      };

      let Some((min, max)) = range(&args[0]) else {
        return Ok(unevaluated("RandomReal", args));
      };

      fn make_random_array(dims: &[usize], min: f64, max: f64) -> Expr {
        let n = dims[0];
        if dims.len() == 1 {
          let results: Vec<Expr> = crate::with_rng(|rng| {
            (0..n)
              .map(|_| {
                Expr::Real(min + rng.random_range(0.0..1.0) * (max - min))
              })
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

/// A `WorkingPrecision` beyond machine precision, in decimal digits.
/// `None` for MachinePrecision or anything at or below it.
fn extended_precision(wp: &Expr) -> Option<f64> {
  let p = try_eval_to_f64(wp)?;
  (p.is_finite() && p > 15.954589770191003).then_some(p)
}

/// The dimensions of a random array (`n` or `{n1, n2, …}`); `Some(vec![])`
/// for a single value. `None` for anything else.
fn random_dims(spec: Option<&Expr>) -> Option<Vec<usize>> {
  match spec {
    None => Some(Vec::new()),
    Some(Expr::Integer(n)) if *n >= 0 => Some(vec![*n as usize]),
    Some(Expr::List(items)) => items
      .iter()
      .map(|i| match i {
        Expr::Integer(n) if *n >= 0 => Some(*n as usize),
        _ => None,
      })
      .collect(),
    _ => None,
  }
}

/// A nested list of the given dimensions filled by `draw` (just one value
/// for no dimensions).
fn fill_with(
  dims: &[usize],
  draw: &mut dyn FnMut() -> Result<Expr, InterpreterError>,
) -> Result<Expr, InterpreterError> {
  let Some((&n, rest)) = dims.split_first() else {
    return draw();
  };
  let mut items = Vec::with_capacity(n);
  for _ in 0..n {
    items.push(fill_with(rest, draw)?);
  }
  Ok(Expr::List(items.into()))
}

/// A uniform draw from [lo, hi] carrying `p` digits of precision: lo plus
/// (hi − lo) times a random fraction with a few more decimal digits than
/// asked for, evaluated by N at precision p.
fn uniform_at_precision(
  lo: &Expr,
  hi: &Expr,
  p: f64,
) -> Result<Expr, InterpreterError> {
  use rand::RngExt;
  let digits = p.ceil() as usize + 5;
  let numerator: String = crate::with_rng(|rng| {
    (0..digits)
      .map(|_| char::from(b'0' + rng.random_range(0..10u8)))
      .collect()
  });
  let numerator = numerator
    .parse::<num_bigint::BigInt>()
    .map_or(Expr::Integer(0), crate::functions::math_ast::bigint_to_expr);
  let fraction = times(vec![
    numerator,
    pow(Expr::Integer(10), Expr::Integer(-(digits as i128))),
  ]);
  let value = plus(vec![
    lo.clone(),
    times(vec![
      plus(vec![hi.clone(), times(vec![Expr::Integer(-1), lo.clone()])]),
      fraction,
    ]),
  ]);
  crate::evaluator::evaluate_expr_to_expr(&call(
    "N",
    vec![value, Expr::Real(p)],
  ))
}

/// Split trailing option rules off a random-number call, returning the
/// positional arguments and the `WorkingPrecision` value if one was given.
fn split_random_options(args: &[Expr]) -> (Vec<Expr>, Option<Expr>) {
  let mut positional = Vec::with_capacity(args.len());
  let mut precision = None;
  for arg in args {
    match arg {
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => {
        if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "WorkingPrecision")
        {
          precision = Some(replacement.as_ref().clone());
        }
      }
      other => positional.push(other.clone()),
    }
  }
  (positional, precision)
}

/// Draw `n` values from a distribution the user defined by giving
/// `Random`DistributionVector[dist, n, precision]` a rule of its own.
/// `None` when there is no such definition, or it did not produce a list of
/// the length asked for.
fn distribution_vector_draw(dist: &Expr, n: i128) -> Option<Expr> {
  let call = call(
    "Random`DistributionVector",
    vec![dist.clone(), Expr::Integer(n), id_expr("MachinePrecision")],
  );
  match crate::evaluator::evaluate_expr_to_expr(&call) {
    Ok(result) => match &result {
      Expr::List(items) if items.len() == n as usize => Some(result),
      _ => None,
    },
    Err(_) => None,
  }
}

/// RandomComplex[] or RandomComplex[max] or RandomComplex[{min, max}]
/// or RandomComplex[range, dims]. Real and imaginary parts are drawn
/// uniformly from the specified rectangle.
pub fn random_complex_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let args = &*integral_dims(args);
  use rand::RngExt;
  // Options (WorkingPrecision) follow the range and dimensions.
  let (args, working_precision) = split_random_options(args);
  let args = args.as_slice();
  if let Some(p) = working_precision.as_ref().and_then(extended_precision)
    && let Some((lo, hi)) = match args.first() {
      None => Some((
        Expr::Integer(0),
        call("Complex", vec![Expr::Integer(1), Expr::Integer(1)]),
      )),
      Some(Expr::List(items)) if items.len() == 2 => {
        Some((items[0].clone(), items[1].clone()))
      }
      Some(max) => Some((Expr::Integer(0), max.clone())),
    }
    && let Some(dims) = random_dims(args.get(1))
  {
    let part = |head: &str, e: &Expr| {
      crate::evaluator::evaluate_expr_to_expr(&call1(head, e.clone()))
    };
    let (re_lo, re_hi) = (part("Re", &lo)?, part("Re", &hi)?);
    let (im_lo, im_hi) = (part("Im", &lo)?, part("Im", &hi)?);
    return fill_with(&dims, &mut || {
      let re = uniform_at_precision(&re_lo, &re_hi, p)?;
      let im = uniform_at_precision(&im_lo, &im_hi, p)?;
      crate::evaluator::evaluate_expr_to_expr(&plus(vec![
        re,
        times(vec![im, id_expr("I")]),
      ]))
    });
  }

  // `(re, im)` of a numeric bound (`-1 - I` is `Complex[-1, -1]`, but a
  // bound may also arrive as an unevaluated sum or product).
  fn complex_parts(expr: &Expr) -> Option<(f64, f64)> {
    crate::functions::list_helpers_ast::expr_to_complex_parts(expr)
  }

  fn make_complex(re: f64, im: f64) -> Expr {
    crate::functions::math_ast::build_complex_float_expr_keep_real(re, im)
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
  // A bound beyond the machine range (10^1000 + 10^1000 I) has no machine
  // sample to draw.
  if ![re_lo, re_hi, im_lo, im_hi].iter().all(|b| b.is_finite()) {
    return Ok(unevaluated("RandomComplex", args));
  }

  fn draw_one(
    rng: &mut dyn rand::Rng,
    re_lo: f64,
    re_hi: f64,
    im_lo: f64,
    im_hi: f64,
  ) -> Expr {
    // Order each corner's bounds so a reversed range (e.g. {2 + 2 I, 0}) is
    // sampled over [0, 2] rather than panicking on an empty range.
    let (re_lo, re_hi) = (re_lo.min(re_hi), re_lo.max(re_hi));
    let (im_lo, im_hi) = (im_lo.min(im_hi), im_lo.max(im_hi));
    let re = if re_lo == re_hi {
      re_lo
    } else {
      rng.random_range(re_lo..re_hi)
    };
    let im = if im_lo == im_hi {
      im_lo
    } else {
      rng.random_range(im_lo..im_hi)
    };
    make_complex(re, im)
  }

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
              _ => return Ok(random_array_dims_error("RandomComplex", args)),
            }
          }
          dims
        }
        _ => return Ok(random_array_dims_error("RandomComplex", args)),
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
    _ => Ok(unevaluated("RandomComplex", args)),
  }
}

/// RandomColor[] returns a single `RGBColor[r, g, b]` whose three channels
/// are independent uniform draws from [0, 1). `RandomColor[n]` returns a
/// list of `n` such colours.
pub fn random_color_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use rand::RngExt;

  fn one_color() -> Expr {
    let (r, g, b) = crate::with_rng(|rng| {
      (
        rng.random_range(0.0..1.0),
        rng.random_range(0.0..1.0),
        rng.random_range(0.0..1.0),
      )
    });
    call(
      "RGBColor",
      vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)],
    )
  }

  // A color template such as RGBColor[_, _, _] or Hue[_, 1, 1]: every
  // blank is filled with a uniform random number.
  fn template_color(model: &Expr) -> Option<Expr> {
    let Expr::FunctionCall { name, args } = model else {
      return None;
    };
    if !matches!(
      name.as_str(),
      "RGBColor" | "Hue" | "GrayLevel" | "CMYKColor" | "XYZColor" | "LABColor"
    ) {
      return None;
    }
    let is_blank = |e: &Expr| match e {
      Expr::Pattern {
        name,
        head: None,
        blank_type: 1,
      } => name.is_empty(),
      Expr::FunctionCall { name, args } => name == "Blank" && args.is_empty(),
      _ => false,
    };
    if !args.iter().any(is_blank) {
      return None;
    }
    let filled: Vec<Expr> = args
      .iter()
      .map(|a| {
        if is_blank(a) {
          Expr::Real(crate::with_rng(|rng| rng.random_range(0.0..1.0)))
        } else {
          a.clone()
        }
      })
      .collect();
    Some(call(name, filled))
  }
  if let Some(model) = args.first()
    && template_color(model).is_some()
  {
    fn fill(model: &Expr, dims: &[usize]) -> Expr {
      match dims.split_first() {
        None => template_color(model).unwrap_or_else(|| model.clone()),
        Some((&n, rest)) => Expr::List(
          (0..n).map(|_| fill(model, rest)).collect::<Vec<_>>().into(),
        ),
      }
    }
    let dims: Option<Vec<usize>> = match args.get(1) {
      None => Some(vec![]),
      Some(Expr::Integer(n)) if *n >= 0 => Some(vec![*n as usize]),
      Some(Expr::List(items)) => items
        .iter()
        .map(|i| match i {
          Expr::Integer(n) if *n >= 0 => Some(*n as usize),
          _ => None,
        })
        .collect(),
      Some(_) => None,
    };
    return match dims {
      Some(dims) if args.len() <= 2 => Ok(fill(model, &dims)),
      _ => Ok(unevaluated("RandomColor", args)),
    };
  }

  match args.len() {
    0 => Ok(one_color()),
    1 => match &args[0] {
      // Wolfram interprets the lone integer argument as a *count* only when
      // it is positive; 0 is read as a color-model specification (which fails
      // with RandomColor::bdmdl, leaving the call unevaluated).
      Expr::Integer(n) if *n > 0 => {
        let count = *n as usize;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
          out.push(one_color());
        }
        Ok(Expr::List(out.into()))
      }
      _ => Ok(unevaluated("RandomColor", args)),
    },
    _ => Ok(unevaluated("RandomColor", args)),
  }
}

/// `RandomDate[]` returns a single `DateObject` at a random instant in the
/// current calendar year (local time). `RandomDate[n]` returns a list of `n`
/// such DateObjects.
#[cfg(not(target_arch = "wasm32"))]
pub fn random_date_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use chrono::{Datelike, Local, TimeZone, Timelike};
  use rand::RngExt;

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
    let offset_ms = crate::with_rng(|rng| rng.random_range(0.0..span));
    let total_micros = (offset_ms * 1000.0) as i64;
    let chosen = year_start + chrono::Duration::microseconds(total_micros);
    let seconds = chosen.second() as f64 + (chosen.nanosecond() as f64) / 1e9;
    let tz_offset_hours = chosen.offset().local_minus_utc() as f64 / 3600.0;
    call(
      "DateObject",
      vec![
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
        str_expr("Instant"),
        str_expr("Gregorian"),
        Expr::Real(tz_offset_hours),
      ],
    )
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
      _ => Ok(unevaluated("RandomDate", args)),
    },
    _ => Ok(unevaluated("RandomDate", args)),
  }
}

#[cfg(target_arch = "wasm32")]
pub fn random_date_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use rand::RngExt;

  fn one_date() -> Expr {
    let now = js_sys::Date::new_0();
    let year = now.get_full_year();
    let year_start = js_sys::Date::new_with_year_month_day(year, 0, 1);
    let next_year_start = js_sys::Date::new_with_year_month_day(year + 1, 0, 1);
    let span = (next_year_start.get_time() - year_start.get_time()).max(1.0);
    let offset_ms = crate::with_rng(|rng| rng.random_range(0.0..span));
    let total_micros = (offset_ms * 1000.0) as i64;
    let chosen = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(
      year_start.get_time() + (total_micros / 1000) as f64,
    ));
    // Local midnight is on a whole-second boundary, so the chosen instant's
    // sub-second part is exactly the offset's microsecond remainder.
    let seconds =
      chosen.get_seconds() as f64 + (total_micros % 1_000_000) as f64 / 1e6;
    let tz_offset_hours = -(chosen.get_timezone_offset() / 60.0);
    call(
      "DateObject",
      vec![
        Expr::List(
          vec![
            Expr::Integer(chosen.get_full_year() as i128),
            Expr::Integer((chosen.get_month() + 1) as i128),
            Expr::Integer(chosen.get_date() as i128),
            Expr::Integer(chosen.get_hours() as i128),
            Expr::Integer(chosen.get_minutes() as i128),
            Expr::Real(seconds),
          ]
          .into(),
        ),
        str_expr("Instant"),
        str_expr("Gregorian"),
        Expr::Real(tz_offset_hours),
      ],
    )
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
      _ => Ok(unevaluated("RandomDate", args)),
    },
    _ => Ok(unevaluated("RandomDate", args)),
  }
}

/// Current local time of day in seconds since midnight (with sub-second
/// precision), used as the anchor for `RandomTime[Quantity[…]]`.
#[cfg(not(target_arch = "wasm32"))]
fn now_time_of_day_seconds() -> f64 {
  use chrono::{Local, Timelike};
  let now = Local::now();
  now.num_seconds_from_midnight() as f64 + now.nanosecond() as f64 / 1e9
}

#[cfg(target_arch = "wasm32")]
fn now_time_of_day_seconds() -> f64 {
  let now = js_sys::Date::new_0();
  now.get_hours() as f64 * 3600.0
    + now.get_minutes() as f64 * 60.0
    + now.get_seconds() as f64
    + now.get_milliseconds() as f64 / 1000.0
}

/// Signed span in seconds of a `Quantity[v, unit]` time quantity.
fn quantity_span_seconds(expr: &Expr) -> Option<f64> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "Quantity" || args.len() != 2 {
    return None;
  }
  let value = try_eval_to_f64(&args[0])?;
  let unit = match &args[1] {
    Expr::String(s) => s.as_str(),
    Expr::Identifier(s) => s.as_str(),
    _ => return None,
  };
  let factor = match unit {
    "Millisecond" | "Milliseconds" => 0.001,
    "Second" | "Seconds" => 1.0,
    "Minute" | "Minutes" => 60.0,
    "Hour" | "Hours" => 3600.0,
    "Day" | "Days" => 86400.0,
    "Week" | "Weeks" => 7.0 * 86400.0,
    _ => return None,
  };
  Some(value * factor)
}

/// The `[start, end)` span in seconds-since-midnight covered by a
/// `TimeObject[{…}, granularity]`. A whole-hour or whole-minute TimeObject
/// covers its full granularity unit; an Instant has a zero-width span.
fn time_object_span(expr: &Expr) -> Option<(f64, f64)> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "TimeObject" || args.is_empty() || args.len() > 2 {
    return None;
  }
  let Expr::List(comps) = &args[0] else {
    return None;
  };
  if comps.is_empty() || comps.len() > 3 {
    return None;
  }
  let mut secs = [0.0f64; 3];
  for (i, c) in comps.iter().enumerate() {
    secs[i] = try_eval_to_f64(c)?;
  }
  let start = (secs[0] * 3600.0 + secs[1] * 60.0 + secs[2]).rem_euclid(86400.0);
  let granularity = match args.get(1) {
    Some(Expr::Identifier(s) | Expr::String(s)) => s.as_str(),
    Some(_) => return None,
    None => match comps.len() {
      1 => "Hour",
      2 => "Minute",
      _ => "Instant",
    },
  };
  let width = match granularity {
    "Hour" => 3600.0,
    "Minute" => 60.0,
    "Second" => 1.0,
    "Instant" => 0.0,
    _ => return None,
  };
  Some((start, start + width))
}

/// `RandomTime[]` returns a single `TimeObject` at a random instant of the
/// day (uniform over the full 24 hours). Other supported forms:
///   * `RandomTime[n]` — list of `n` such TimeObjects
///   * `RandomTime[Quantity[q, unit]]` — instant within `q` of the current
///     time (before it for negative `q`)
///   * `RandomTime[t]` — instant within the span of TimeObject `t`
///   * `RandomTime[{t1, t2}]` — instant between `t1` and `t2` (covering the
///     end TimeObject's full granularity span, like wolframscript)
///   * `RandomTime[spec, n]` — list of `n` draws from any spec above
pub fn random_time_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use rand::RngExt;

  fn build_time(total: f64) -> Expr {
    // Round to microsecond resolution (the same precision RandomDate uses
    // for its seconds field).
    let total_micros = (total.rem_euclid(86400.0) * 1e6) as i64;
    let h = total_micros / 3_600_000_000;
    let m = (total_micros % 3_600_000_000) / 60_000_000;
    let s = (total_micros % 60_000_000) as f64 / 1e6;
    call(
      "TimeObject",
      vec![
        Expr::List(
          vec![
            Expr::Integer(h as i128),
            Expr::Integer(m as i128),
            Expr::Real(s),
          ]
          .into(),
        ),
        str_expr("Instant"),
      ],
    )
  }

  fn sample(lo: f64, hi: f64) -> Expr {
    if hi > lo {
      build_time(crate::with_rng(|rng| rng.random_range(lo..hi)))
    } else {
      build_time(lo)
    }
  }

  // The `[lo, hi)` sampling interval described by a spec argument, or None
  // when the spec isn't one of the supported forms.
  fn spec_interval(spec: &Expr) -> Option<(f64, f64)> {
    if let Some(span) = quantity_span_seconds(spec) {
      // A time quantity anchors at the current time: within the next `span`
      // seconds, or the previous ones for a negative quantity.
      let now = now_time_of_day_seconds();
      return Some(if span >= 0.0 {
        (now, now + span)
      } else {
        (now + span, now)
      });
    }
    if let Some((start, end)) = time_object_span(spec) {
      return Some((start, end));
    }
    if let Expr::List(items) = spec
      && items.len() == 2
      && let Some((s1, _)) = time_object_span(&items[0])
      && let Some((s2, e2)) = time_object_span(&items[1])
    {
      // In-order bounds cover both spans; reversed bounds cover the times
      // between the two spans (matching wolframscript).
      return Some(if s1 <= s2 { (s1, e2) } else { (e2, s1) });
    }
    None
  }

  let unevaluated = || Ok(unevaluated("RandomTime", args));

  // Split the arguments into an optional spec and an optional count.
  let (interval, count): (Option<(f64, f64)>, Option<i128>) = match args {
    [] => (None, None),
    [Expr::Integer(n)] => (None, Some(*n)),
    [spec] => match spec_interval(spec) {
      Some(iv) => (Some(iv), None),
      None => return unevaluated(),
    },
    [spec, Expr::Integer(n)] => match spec_interval(spec) {
      Some(iv) => (Some(iv), Some(*n)),
      None => return unevaluated(),
    },
    _ => return unevaluated(),
  };

  let (lo, hi) = interval.unwrap_or((0.0, 86400.0));
  match count {
    None => Ok(sample(lo, hi)),
    Some(n) if n >= 0 => {
      let count = n as usize;
      let mut out = Vec::with_capacity(count);
      for _ in 0..count {
        out.push(sample(lo, hi));
      }
      Ok(Expr::List(out.into()))
    }
    Some(_) => unevaluated(),
  }
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
    _ => Ok(unevaluated("Random", args)),
  }
}

/// RandomChoice[list]
pub fn random_choice_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let args = &*integral_dims(args);
  use rand::RngExt;

  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "RandomChoice expects 1 or 2 arguments".into(),
    ));
  }

  // A span of integers `a ;; b` or `a ;; b ;; step` is the list
  // `Range[a, b, step]` to choose from.
  if let Expr::FunctionCall { name, args: span } = &args[0]
    && name == "Span"
    && (span.len() == 2 || span.len() == 3)
    && span.iter().all(|e| matches!(e, Expr::Integer(_)))
  {
    let range = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Range".to_string(),
      args: span.clone(),
    })?;
    if matches!(&range, Expr::List(items) if !items.is_empty()) {
      let mut new_args = args.to_vec();
      new_args[0] = range;
      return random_choice_ast(&new_args);
    }
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
        // Every malformed rule leaves the call unevaluated with the
        // message wolframscript uses for it: a non-list (or empty) choice
        // side is `lrwl`, mismatched lengths `wghtvl`, and anything but
        // non-negative real weights with a positive total `wghtv`.
        let rule = expr_to_string(&args[0]);
        let reject = |tag: &str, text: &str| {
          crate::emit_message(&format!("RandomChoice::{tag}: {text}"));
          Ok(unevaluated("RandomChoice", args))
        };
        let lrwl = || {
          reject(
            "lrwl",
            &format!(
              "The items for choice {rule} should be a nonempty list or a \
               rule weights -> choices."
            ),
          )
        };
        let wghtv = || {
          reject(
            "wghtv",
            &format!(
              "The weights given on the left-hand side of {rule} should be \
               a list of positive numerical quantities having the same \
               length as the list given on the right-hand side."
            ),
          )
        };
        let Expr::List(values) = replacement.as_ref() else {
          return lrwl();
        };
        let Expr::List(weights) = pattern.as_ref() else {
          return wghtv();
        };
        if weights.is_empty() && values.is_empty() {
          return lrwl();
        }
        if weights.len() != values.len() {
          return reject(
            "wghtvl",
            &format!(
              "The weights given on the left-hand side of {rule} should \
               have the same length as the data given on the right-hand \
               side."
            ),
          );
        }
        let mut cdf = Vec::with_capacity(weights.len());
        let mut acc = 0.0f64;
        for w in weights {
          match try_eval_to_f64(w) {
            Some(v) if v >= 0.0 => {
              acc += v;
              cdf.push(acc);
            }
            _ => return wghtv(),
          }
        }
        if acc <= 0.0 {
          return wghtv();
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
        // An empty choice list emits lrwl and stays unevaluated rather than
        // raising a hard error (wolframscript parity).
        crate::emit_message(&format!(
          "RandomChoice::lrwl: The items for choice {} should be a nonempty list or a rule weights -> choices.",
          expr_to_string(&args[0])
        ));
        return Ok(unevaluated("RandomChoice", args));
      }
      _ => {
        return Ok(unevaluated("RandomChoice", args));
      }
    }
  };
  // Helper: pick one index given the (optional) weight CDF.
  let pick_index = |rng: &mut dyn rand::Rng| -> usize {
    if let Some(cdf) = weighted_cdf.as_ref() {
      let total = *cdf.last().unwrap();
      let u = rng.random_range(0.0..total);
      cdf.iter().position(|&c| u < c).unwrap_or(cdf.len() - 1)
    } else {
      rng.random_range(0..items.len())
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
        for d in ds {
          match d {
            Expr::Integer(k) if *k >= 0 => out.push(*k as usize),
            _ => return Ok(random_array_dims_error("RandomChoice", args)),
          }
        }
        out
      }
      _ => return Ok(random_array_dims_error("RandomChoice", args)),
    };
    fn build(items: &[Expr], dims: &[usize], cdf: Option<&[f64]>) -> Expr {
      use rand::RngExt;
      let n = dims[0];
      if dims.len() == 1 {
        let result: Vec<Expr> = crate::with_rng(|rng| {
          (0..n)
            .map(|_| {
              let idx = if let Some(cdf) = cdf {
                let total = *cdf.last().unwrap();
                let u = rng.random_range(0.0..total);
                cdf.iter().position(|&c| u < c).unwrap_or(cdf.len() - 1)
              } else {
                rng.random_range(0..items.len())
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
  let args = &*integral_dims(args);
  use rand::RngExt;
  use rand::seq::{IndexedRandom, SliceRandom};

  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "RandomSample expects 1 or 2 arguments".into(),
    ));
  }

  // RandomSample[<|k1 -> v1, ...|>, n] samples whole key-value pairs and
  // returns them as an Association.
  if let Expr::Association(pairs) = &args[0] {
    let mut rest = args.to_vec();
    rest[0] = Expr::List(
      pairs
        .iter()
        .map(|(k, v)| Expr::Rule {
          pattern: Box::new(k.clone()),
          replacement: Box::new(v.clone()),
        })
        .collect::<Vec<_>>()
        .into(),
    );
    let sampled_expr = random_sample_ast(&rest)?;
    return match &sampled_expr {
      Expr::List(sampled) => {
        let mut out = Vec::with_capacity(sampled.len());
        for item in sampled {
          match item {
            Expr::Rule {
              pattern,
              replacement,
            } => {
              out
                .push((pattern.as_ref().clone(), replacement.as_ref().clone()));
            }
            _ => return Ok(unevaluated("RandomSample", args)),
          }
        }
        Ok(Expr::Association(out))
      }
      _ => Ok(unevaluated("RandomSample", args)),
    };
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
        return Ok(unevaluated("RandomSample", args));
      };
      if weights.is_empty() || weights.len() != values.len() {
        return Err(InterpreterError::EvaluationError(
          "RandomSample: weight and value lists must have matching length"
            .into(),
        ));
      }
      let mut ws = Vec::with_capacity(weights.len());
      for w in weights {
        let v = try_eval_to_f64(w).ok_or_else(|| {
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
        return Ok(unevaluated("RandomSample", args));
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
            let u: f64 = rng.random_range(f64::MIN_POSITIVE..1.0);
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
        let arg_strs: Vec<String> = args.iter().map(expr_to_output).collect();
        crate::emit_message(&format!(
          "RandomSample::intnm: Non-negative machine-sized integer expected at position 2 in RandomSample[{}].",
          arg_strs.join(", "),
        ));
        return Ok(unevaluated("RandomSample", args));
      }
    };
    if n > items.len() {
      crate::emit_message(&format!(
        "RandomSample::smplen: RandomSample cannot generate a sample of \
         length {}, which is greater than the length of the sample set {}. If \
         you want a choice of possibly repeated elements from the set, use \
         RandomChoice.",
        n,
        crate::syntax::format_expr(&args[0], crate::syntax::ExprForm::Output)
      ));
      return Ok(unevaluated("RandomSample", args));
    }
    if let Some(ws) = weights.as_ref() {
      // Pick the n items with the largest Efraimidis–Spirakis keys.
      let keys: Vec<f64> = crate::with_rng(|rng| {
        ws.iter()
          .map(|&w| {
            let u: f64 = rng.random_range(f64::MIN_POSITIVE..1.0);
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
        crate::with_rng(|rng| items.sample(rng, n).cloned().collect());
      Ok(Expr::List(sampled.into()))
    }
  }
}

/// Build a single random permutation of the points `1..=n` (via a
/// Fisher–Yates shuffle) and return it in canonical `Cycles[{...}]` form:
/// fixed points are dropped, each cycle is rotated to start with its
/// smallest element, and cycles are sorted by that first element.
fn random_permutation_cycles(n: usize) -> Expr {
  use rand::seq::SliceRandom;

  // perm[i] is the image of point (i + 1) under the permutation.
  let mut perm: Vec<usize> = (1..=n).collect();
  crate::with_rng(|rng| perm.shuffle(rng));

  // Decompose the permutation into disjoint cycles.
  let mut visited = vec![false; n + 1];
  let mut cycles: Vec<Vec<i128>> = Vec::new();
  for start in 1..=n {
    if visited[start] || perm[start - 1] == start {
      visited[start] = true;
      continue;
    }
    let mut cycle: Vec<i128> = Vec::new();
    let mut j = start;
    while !visited[j] {
      visited[j] = true;
      cycle.push(j as i128);
      j = perm[j - 1];
    }
    if cycle.len() >= 2 {
      // Rotate so the smallest element comes first.
      let (min_idx, _) =
        cycle.iter().enumerate().min_by_key(|(_, v)| **v).unwrap();
      let mut rotated = cycle[min_idx..].to_vec();
      rotated.extend_from_slice(&cycle[..min_idx]);
      cycles.push(rotated);
    }
  }
  // Sort cycles by their first element.
  cycles.sort_by_key(|c| c[0]);

  let cycle_exprs: Vec<Expr> = cycles
    .into_iter()
    .map(|c| {
      Expr::List(c.into_iter().map(Expr::Integer).collect::<Vec<_>>().into())
    })
    .collect();
  call1("Cycles", Expr::List(cycle_exprs.into()))
}

/// RandomPermutation[n] — a random permutation of `n` points as a `Cycles`
/// object. RandomPermutation[n, k] — a list of `k` such permutations.
///
/// A non-integer first argument (e.g. a list like `{4}`) is not a valid point
/// count or permutation group, so the call stays unevaluated — matching
/// wolframscript, which emits `RandomPermutation::grp` and returns the input.
pub fn random_permutation_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "RandomPermutation expects 1 or 2 arguments".into(),
    ));
  }

  // First argument: number of points. Only a non-negative integer is valid.
  let n = match &args[0] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(unevaluated("RandomPermutation", args));
    }
  };

  if args.len() == 1 {
    Ok(random_permutation_cycles(n))
  } else {
    let k = match &args[1] {
      Expr::Integer(k) if *k >= 0 => *k as usize,
      _ => {
        return Ok(unevaluated("RandomPermutation", args));
      }
    };
    let perms: Vec<Expr> =
      (0..k).map(|_| random_permutation_cycles(n)).collect();
    Ok(Expr::List(perms.into()))
  }
}

/// Whether `expr` holds an `==` anywhere.
fn contains_equality(expr: &Expr) -> bool {
  match expr {
    Expr::Comparison { operators, .. }
      if operators.contains(&crate::syntax::ComparisonOp::Equal) =>
    {
      true
    }
    Expr::FunctionCall { name, .. } if name == "Equal" => true,
    _ => crate::syntax::expr_children(expr)
      .into_iter()
      .any(contains_equality),
  }
}

/// Cumulative weight table for sampling `ProbabilityDistribution[f, range]`.
struct SamplingTable {
  /// Left edges of the grid cells (continuous) or the support points.
  points: Vec<f64>,
  cumulative: Vec<f64>,
  total: f64,
  /// Cell width for a continuous distribution, `None` for a discrete one.
  width: Option<f64>,
}

fn probability_distribution_table(dargs: &[Expr]) -> Option<SamplingTable> {
  use crate::functions::plot::substitute_var;
  const CELLS: usize = 4000;
  let [pdf, Expr::List(range)] = dargs else {
    return None;
  };
  let Expr::Identifier(var) = range.first()? else {
    return None;
  };
  let bound = |e: &Expr| -> Option<f64> {
    match e {
      Expr::Identifier(s) if s == "Infinity" => Some(f64::INFINITY),
      _ => {
        let v = crate::evaluator::evaluate_expr_to_expr(e).ok()?;
        if matches!(&v, Expr::Identifier(s) if s == "Infinity") {
          Some(f64::INFINITY)
        } else if crate::functions::math_ast::is_neg_infinity(&v) {
          Some(f64::NEG_INFINITY)
        } else {
          try_eval_to_f64(&v)
        }
      }
    }
  };
  let mut lo = bound(range.get(1)?)?;
  let mut hi = bound(range.get(2)?)?;
  if lo.is_nan() || hi.is_nan() || hi < lo {
    return None;
  }
  let weight_at = |x: f64| -> f64 {
    let e = substitute_var(pdf, var, &Expr::Real(x));
    crate::evaluator::evaluate_expr_to_expr(&e)
      .ok()
      .and_then(|v| try_eval_to_f64(&v))
      .filter(|w| w.is_finite() && *w > 0.0)
      .unwrap_or(0.0)
  };
  // An infinite bound of a continuous density is replaced by where its
  // tail becomes negligible: the density is scanned on a log-spaced grid
  // (±10^-3 … ±10^7 around 0, inside the finite bound if any) and cut where
  // it falls below 10^-14 of its peak.
  if range.len() == 3 && !(lo.is_finite() && hi.is_finite()) {
    let mut scan: Vec<f64> = vec![0.0];
    for k in 0..=40 {
      let r = 10f64.powf(-3.0 + 0.25 * k as f64);
      scan.push(r);
      scan.push(-r);
    }
    if lo.is_finite() {
      scan.push(lo);
    }
    if hi.is_finite() {
      scan.push(hi);
    }
    scan.retain(|x| *x >= lo && *x <= hi);
    scan.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let weights: Vec<f64> = scan.iter().map(|&x| weight_at(x)).collect();
    let peak = weights.iter().copied().fold(0.0_f64, f64::max);
    if peak <= 0.0 {
      return None;
    }
    let first = weights.iter().position(|w| *w > peak * 1e-14)?;
    let last = weights.iter().rposition(|w| *w > peak * 1e-14)?;
    if !lo.is_finite() {
      lo = scan[first.saturating_sub(1)];
    }
    if !hi.is_finite() {
      hi = scan[(last + 1).min(scan.len() - 1)];
    }
    if hi.partial_cmp(&lo) != Some(std::cmp::Ordering::Greater) {
      return None;
    }
  }
  if !lo.is_finite() || !hi.is_finite() {
    return None;
  }
  let (points, weights, width) = match range.len() {
    3 => {
      let w = (hi - lo) / CELLS as f64;
      let points: Vec<f64> = (0..CELLS).map(|i| lo + w * i as f64).collect();
      let weights: Vec<f64> =
        points.iter().map(|p| weight_at(p + w / 2.0)).collect();
      (points, weights, Some(w))
    }
    4 => {
      let step = try_eval_to_f64(&range[3])?;
      if step <= 0.0 {
        return None;
      }
      let count = ((hi - lo) / step + 1e-9).floor() as usize + 1;
      let points: Vec<f64> = (0..count).map(|i| lo + step * i as f64).collect();
      let weights = points.iter().map(|&p| weight_at(p)).collect();
      (points, weights, None)
    }
    _ => return None,
  };
  let mut total = 0.0;
  let cumulative: Vec<f64> = weights
    .iter()
    .map(|w: &f64| {
      total += w;
      total
    })
    .collect();
  if total <= 0.0 {
    return None;
  }
  Some(SamplingTable {
    points,
    cumulative,
    total,
    width,
  })
}

/// A closure drawing one matrix from a Gaussian random-matrix ensemble, or
/// (for `MatrixPropertyDistribution[prop, Distributed[x, ensemble]]`) one
/// matrix of the underlying ensemble. `None` when `name`/`dargs` is not such
/// a distribution with numeric parameters.
fn matrix_ensemble_sampler<'a>(
  name: &str,
  dargs: &'a [Expr],
) -> Option<Box<dyn Fn() -> Expr + 'a>> {
  if name == "MatrixPropertyDistribution" {
    let [
      _,
      Expr::FunctionCall {
        name: dname,
        args: dspec,
      },
    ] = dargs
    else {
      return None;
    };
    if dname != "Distributed"
      || dspec.len() != 2
      || !matches!(dspec[0], Expr::Identifier(_))
    {
      return None;
    }
    let Expr::FunctionCall {
      name: ename,
      args: eargs,
    } = &dspec[1]
    else {
      return None;
    };
    return matrix_ensemble_sampler(ename, eargs);
  }
  let unitary = match name {
    "GaussianOrthogonalMatrixDistribution" => false,
    "GaussianUnitaryMatrixDistribution" => true,
    _ => return None,
  };
  let (sigma, size) = match dargs {
    [size] => (1.0, size),
    [sigma, size] => (try_eval_to_f64(sigma)?, size),
    _ => return None,
  };
  if sigma <= 0.0 {
    return None;
  }
  let size = match size {
    Expr::Integer(k) if *k > 0 => *k as usize,
    _ => return None,
  };
  Some(Box::new(move || {
    use rand_distr::{Distribution, Normal};
    // Density ∝ exp(-Tr(m²) / (2σ²)): real diagonal with variance σ², and
    // off-diagonal entries (each real/imaginary part) with variance σ²/2.
    let diag = Normal::new(0.0, sigma).unwrap();
    let off = Normal::new(0.0, sigma / std::f64::consts::SQRT_2).unwrap();
    let mut entries = vec![vec![(0.0f64, 0.0f64); size]; size];
    crate::with_rng(|rng| {
      for i in 0..size {
        entries[i][i] = (diag.sample(rng), 0.0);
        for j in i + 1..size {
          let re = off.sample(rng);
          let im = if unitary { off.sample(rng) } else { 0.0 };
          entries[i][j] = (re, im);
          entries[j][i] = (re, -im);
        }
      }
    });
    Expr::List(
      entries
        .into_iter()
        .map(|row| {
          Expr::List(
            row
              .into_iter()
              .map(|(re, im)| {
                if im == 0.0 {
                  Expr::Real(re)
                } else {
                  crate::functions::math_ast::build_complex_float_expr_keep_real(
                    re, im,
                  )
                }
              })
              .collect::<Vec<_>>()
              .into(),
          )
        })
        .collect::<Vec<_>>()
        .into(),
    )
  }))
}

/// RandomVariate[dist] or RandomVariate[dist, n]
/// Supports UniformDistribution[{min, max}] and NormalDistribution[mu, sigma]
/// A distribution with a parameter that is no number (or a scale that is
/// not positive) cannot be sampled: report the first such parameter the
/// way wolframscript does — under `RandomVariate` for some distributions
/// and the distribution's own name for others — and leave the call
/// unevaluated. Parameters that must be positive are checked before
/// location parameters.
fn reject_symbolic_variate_params(dist: &Expr) -> bool {
  let Expr::FunctionCall { name, args } = dist else {
    return false;
  };
  let numeric = |e: &Expr| try_eval_to_f64(e).is_some();
  let head = match name.as_str() {
    "NormalDistribution"
    | "ExponentialDistribution"
    | "PoissonDistribution"
    | "BetaDistribution"
    | "ChiSquareDistribution"
    | "StudentTDistribution" => "RandomVariate",
    other => other,
  };
  let report = |tag: &str, pos: usize, what: &str| {
    crate::emit_message(&format!(
      "{head}::{tag}: Parameter {} at position {pos} in {} is expected to be {what}.",
      expr_to_string(&args[pos - 1]),
      expr_to_string(dist),
    ));
    true
  };
  for &pos in
    crate::functions::math_ast::distributions::positive_param_positions_for(
      name,
      args.len(),
    )
  {
    if args
      .get(pos - 1)
      .is_some_and(|a| try_eval_to_f64(a).is_none_or(|v| v <= 0.0))
    {
      return report("posprm", pos, "positive");
    }
  }
  match name.as_str() {
    "NormalDistribution" | "LogNormalDistribution" | "CauchyDistribution"
      if args.len() == 2 && !numeric(&args[0]) =>
    {
      report("realprm", 1, "real")
    }
    "BinomialDistribution" if args.len() == 2 && !numeric(&args[0]) => {
      report("nnintprm", 1, "a non-negative integer")
    }
    "BinomialDistribution" if args.len() == 2 && !numeric(&args[1]) => {
      report("probprm", 2, "between 0 and 1 inclusive")
    }
    "BernoulliDistribution" if args.len() == 1 && !numeric(&args[0]) => {
      report("probprm", 1, "between 0 and 1 inclusive")
    }
    "UniformDistribution" if matches!(args.as_slice(), [Expr::List(b)] if b.len() == 2 && !b.iter().all(numeric)) =>
    {
      crate::emit_message(&format!(
        "RandomVariate::unifr: -- Message text not found -- ({})",
        expr_to_string(dist)
      ));
      true
    }
    _ => false,
  }
}

pub fn random_variate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let args = &*integral_dims(args);
  if let Some(dist) = args.first()
    && (reject_symbolic_variate_params(dist)
      || crate::functions::math_ast::distributions::reject_bad_distribution_params(
        dist,
      ))
  {
    return Ok(unevaluated("RandomVariate", args));
  }
  use rand::RngExt;
  use rand_distr::{Distribution, Normal};

  let dist = &args[0];
  // A single variate of QuantityDistribution[d, unit] is a variate of d in
  // that unit.
  if args.len() == 1
    && let Expr::FunctionCall { name, args: dargs } = dist
    && name == "QuantityDistribution"
    && dargs.len() == 2
  {
    let value = random_variate_ast(&dargs[..1])?;
    if matches!(&value, Expr::FunctionCall { name, .. } if name == "RandomVariate")
    {
      return Ok(unevaluated("RandomVariate", args));
    }
    return crate::evaluator::evaluate_expr_to_expr(&call(
      "Quantity",
      vec![value, dargs[1].clone()],
    ));
  }
  let n = if args.len() == 2 {
    match &args[1] {
      // A zero count yields an empty list (matching wolframscript).
      Expr::Integer(0) => return Ok(Expr::List(vec![].into())),
      Expr::Integer(n) if *n > 0 => Some(*n as usize),
      // A dimension list {d1, d2, …} produces a nested array of that shape:
      // draw prod(d) samples flat, then reshape.
      Expr::List(dims_expr) => {
        let mut dims = Vec::with_capacity(dims_expr.len());
        for d in dims_expr {
          match d {
            Expr::Integer(k) if *k >= 0 => dims.push(*k as usize),
            _ => return Ok(random_array_dims_error("RandomVariate", args)),
          }
        }
        // Dimensions whose product overflows can never be allocated.
        let Some(total) =
          dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
        else {
          return Ok(random_array_dims_error("RandomVariate", args));
        };
        // A zero total means some dimension is 0; build the empty nested
        // shape directly (ArrayReshape does not reduce zero-size specs).
        if total == 0 {
          fn empty_nested(dims: &[usize]) -> Expr {
            match dims {
              [_] | [] => Expr::List(vec![].into()),
              [n, rest @ ..] => {
                Expr::List((0..*n).map(|_| empty_nested(rest)).collect())
              }
            }
          }
          return Ok(empty_nested(&dims));
        }
        // Draw `total` samples and nest them into `dims` shape by grouping,
        // never by flattening: a vector-valued distribution (Multinormal,
        // Binormal, …) draws whole vectors, and reshaping their flattened
        // reals would scramble those vectors across the wrong dimensions.
        // `RandomVariate[dist, {n}]` for such a distribution must match
        // `RandomVariate[dist, n]`, appending the vector's own length as an
        // implicit trailing dimension — exactly what treating each draw as
        // one opaque leaf achieves.
        let sample =
          random_variate_ast(&[dist.clone(), Expr::Integer(total as i128)])?;
        let Expr::List(flat) = &sample else {
          return Ok(random_array_dims_error("RandomVariate", args));
        };
        fn nest_leaves(
          leaves: &mut impl Iterator<Item = Expr>,
          dims: &[usize],
        ) -> Expr {
          match dims {
            [] => leaves.next().expect("dims product matches leaf count"),
            [d, rest @ ..] => {
              Expr::List((0..*d).map(|_| nest_leaves(leaves, rest)).collect())
            }
          }
        }
        let mut leaves = flat.iter().cloned();
        return Ok(nest_leaves(&mut leaves, &dims));
      }
      _ => return Ok(random_array_dims_error("RandomVariate", args)),
    }
  } else {
    None
  };

  // Random-matrix ensembles and properties derived from them.
  if let Expr::FunctionCall { name, args: dargs } = dist
    && let Some(draw) = matrix_ensemble_sampler(name, dargs)
  {
    let sample = || -> Result<Expr, InterpreterError> {
      let matrix = draw();
      // MatrixPropertyDistribution[prop, Distributed[x, ensemble]]: bind the
      // sampled matrix to x and evaluate the property.
      if let [property, Expr::FunctionCall { args: spec, .. }] =
        dargs.as_slice()
        && name == "MatrixPropertyDistribution"
        && let Expr::Identifier(var) = &spec[0]
      {
        let bound = crate::syntax::substitute_variable(property, var, &matrix);
        return crate::evaluator::evaluate_expr_to_expr(&bound);
      }
      Ok(matrix)
    };
    return match n {
      None => sample(),
      Some(count) => Ok(Expr::List(
        (0..count)
          .map(|_| sample())
          .collect::<Result<Vec<_>, _>>()?
          .into(),
      )),
    };
  }

  // OrderDistribution[{dist, n}, k]: the k-th smallest of n draws of dist.
  if let Expr::FunctionCall { name, args: dargs } = dist
    && name == "OrderDistribution"
    && let [Expr::List(spec), Expr::Integer(k)] = dargs.as_slice()
    && let [base, Expr::Integer(m)] = spec.as_slice()
    && *k >= 1
    && *k <= *m
  {
    let (m, k) = (*m, *k as usize);
    let one = || -> Result<Option<Expr>, InterpreterError> {
      let sample = random_variate_ast(&[base.clone(), Expr::Integer(m)])?;
      let Expr::List(ref draws) = sample else {
        return Ok(None);
      };
      let mut values: Vec<(f64, Expr)> = Vec::with_capacity(draws.len());
      for d in draws {
        let Some(v) = crate::functions::math_ast::try_eval_to_f64(d) else {
          return Ok(None);
        };
        values.push((v, d.clone()));
      }
      values.sort_by(|a, b| a.0.total_cmp(&b.0));
      Ok(values.get(k - 1).map(|p| p.1.clone()))
    };
    let draw_all =
      |count: usize| -> Result<Option<Vec<Expr>>, InterpreterError> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
          match one()? {
            Some(v) => out.push(v),
            None => return Ok(None),
          }
        }
        Ok(Some(out))
      };
    match n {
      None => {
        if let Some(v) = one()? {
          return Ok(v);
        }
      }
      Some(count) => {
        if let Some(vs) = draw_all(count)? {
          return Ok(Expr::List(vs.into()));
        }
      }
    }
    return Ok(unevaluated("RandomVariate", args));
  }

  // Local helpers for the distributions added below.
  fn sample_poisson(lambda: f64) -> Result<i128, InterpreterError> {
    use rand_distr::{Distribution, Poisson};
    let p = Poisson::new(lambda).map_err(|e| {
      InterpreterError::EvaluationError(format!("PoissonDistribution: {e}"))
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
            let lo = try_eval_to_f64(&bounds[0]).ok_or_else(|| {
              InterpreterError::EvaluationError(
                "UniformDistribution: invalid min bound".into(),
              )
            })?;
            let hi = try_eval_to_f64(&bounds[1]).ok_or_else(|| {
              InterpreterError::EvaluationError(
                "UniformDistribution: invalid max bound".into(),
              )
            })?;
            // Bounds given the wrong way round still name the same
            // interval, and a degenerate one has a single possible value —
            // sampling either without care would panic on an empty range.
            let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
            let draw = |rng: &mut dyn rand::Rng| -> f64 {
              if lo == hi {
                lo
              } else {
                rng.random_range(lo..hi)
              }
            };
            match n {
              None => Ok(Expr::Real(crate::with_rng(|rng| draw(rng)))),
              Some(count) => {
                let results: Vec<Expr> = crate::with_rng(|rng| {
                  (0..count).map(|_| Expr::Real(draw(rng))).collect()
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
          let mu = try_eval_to_f64(&dargs[0]).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "NormalDistribution: invalid mean".into(),
            )
          })?;
          let sigma = try_eval_to_f64(&dargs[1]).ok_or_else(|| {
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
        InterpreterError::EvaluationError(format!("NormalDistribution: {e}"))
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
      if name == "LogNormalDistribution" && dargs.len() == 2 =>
    {
      // exp(X) with X ~ Normal(mu, sigma) is log-normal.
      let (Some(mu), Some(sigma)) =
        (try_eval_to_f64(&dargs[0]), try_eval_to_f64(&dargs[1]))
      else {
        return Ok(unevaluated("RandomVariate", args));
      };
      let Ok(normal) = Normal::new(mu, sigma) else {
        return Ok(unevaluated("RandomVariate", args));
      };
      if sigma <= 0.0 {
        return Ok(unevaluated("RandomVariate", args));
      }
      let sample = || -> Expr {
        Expr::Real(crate::with_rng(|rng| normal.sample(rng)).exp())
      };
      match n {
        None => Ok(sample()),
        Some(count) => Ok(Expr::List(
          (0..count).map(|_| sample()).collect::<Vec<_>>().into(),
        )),
      }
    }
    Expr::FunctionCall { name, args: dargs }
      if (name == "GammaDistribution" && dargs.len() == 2)
        || (name == "WeibullDistribution"
          && (dargs.len() == 2 || dargs.len() == 3)) =>
    {
      // Both take a shape α and a scale β, and both need them positive;
      // WeibullDistribution[α, β, μ] is shifted by its location μ.
      let (Some(alpha), Some(beta)) =
        (try_eval_to_f64(&dargs[0]), try_eval_to_f64(&dargs[1]))
      else {
        return Ok(unevaluated("RandomVariate", args));
      };
      let location = match dargs.get(2) {
        None => 0.0,
        Some(mu) => match try_eval_to_f64(mu) {
          Some(mu) => mu,
          None => return Ok(unevaluated("RandomVariate", args)),
        },
      };
      if !(alpha > 0.0 && beta > 0.0) {
        return Ok(unevaluated("RandomVariate", args));
      }
      let sample: Box<dyn Fn() -> Expr> = if name == "GammaDistribution" {
        let Ok(gamma) = rand_distr::Gamma::new(alpha, beta) else {
          return Ok(unevaluated("RandomVariate", args));
        };
        Box::new(move || {
          Expr::Real(crate::with_rng(|rng| {
            rand_distr::Distribution::sample(&gamma, rng)
          }))
        })
      } else {
        // `rand_distr::Weibull::new` takes (scale, shape).
        let Ok(weibull) = rand_distr::Weibull::new(beta, alpha) else {
          return Ok(unevaluated("RandomVariate", args));
        };
        Box::new(move || {
          Expr::Real(
            location
              + crate::with_rng(|rng| {
                rand_distr::Distribution::sample(&weibull, rng)
              }),
          )
        })
      };
      match n {
        None => Ok(sample()),
        Some(count) => Ok(Expr::List(
          (0..count).map(|_| sample()).collect::<Vec<_>>().into(),
        )),
      }
    }
    Expr::FunctionCall { name, args: dargs }
      if name == "ProbabilityDistribution" && dargs.len() == 2 =>
    {
      // Tabulate the density (continuous `{x, lo, hi}`) on a fine grid, or
      // the mass function (discrete `{x, lo, hi, step}`) at its support
      // points, and draw by inverse transform on the cumulative weights.
      //
      // wolframscript cannot sample a discrete distribution whose mass is a
      // Piecewise of point conditions (`x == 1 || x == 3`) and says so.
      if let Expr::List(range) = &dargs[1]
        && range.len() == 4
        && let Expr::FunctionCall {
          name: pw,
          args: pw_args,
        } = &dargs[0]
        && pw == "Piecewise"
        && contains_equality(pw_args.first().unwrap_or(&dargs[0]))
      {
        crate::emit_message(&format!(
          "RandomVariate::noimp: Sampling from {} is not implemented.",
          crate::syntax::format_expr(&args[0], crate::syntax::ExprForm::Output)
        ));
        return Ok(unevaluated("RandomVariate", args));
      }
      let Some(table) = probability_distribution_table(dargs) else {
        return Ok(unevaluated("RandomVariate", args));
      };
      // A discrete support on an exact grid gives exact values.
      let exact_grid = match &dargs[1] {
        Expr::List(range) if range.len() == 4 => {
          let exact = |e: &Expr| {
            matches!(e, Expr::Integer(_))
              || matches!(e, Expr::FunctionCall { name, .. } if name == "Rational")
          };
          (exact(&range[1]) && exact(&range[3]))
            .then(|| (range[1].clone(), range[3].clone()))
        }
        _ => None,
      };
      let sample = || -> Expr {
        let u: f64 = crate::with_rng(|rng| rng.random_range(0.0..table.total));
        let i = table
          .cumulative
          .partition_point(|&c| c <= u)
          .min(table.cumulative.len() - 1);
        if let Some((lo, step)) = &exact_grid {
          let point = crate::helpers::plus2(
            lo.clone(),
            crate::helpers::times2(Expr::Integer(i as i128), step.clone()),
          );
          return crate::evaluator::evaluate_expr_to_expr(&point)
            .unwrap_or(point);
        }
        let x = match table.width {
          // Continuous: uniform within the selected grid cell.
          Some(w) => {
            table.points[i]
              + w * crate::with_rng(|rng| rng.random_range(0.0..1.0))
          }
          None => table.points[i],
        };
        Expr::Real(x)
      };
      match n {
        None => Ok(sample()),
        Some(count) => Ok(Expr::List(
          (0..count).map(|_| sample()).collect::<Vec<_>>().into(),
        )),
      }
    }
    Expr::FunctionCall { name, args: dargs }
      if name == "MixtureDistribution" && dargs.len() == 2 =>
    {
      // Pick a component by its (normalized) weight, then delegate the draw
      // to that component's own sampler.
      let (Expr::List(weights), Expr::List(comps)) = (&dargs[0], &dargs[1])
      else {
        return Ok(unevaluated("RandomVariate", args));
      };
      let ws: Option<Vec<f64>> = weights.iter().map(try_eval_to_f64).collect();
      let Some(ws) = ws else {
        return Ok(unevaluated("RandomVariate", args));
      };
      let total: f64 = ws.iter().sum();
      if ws.len() != comps.len()
        || ws.is_empty()
        || ws.iter().any(|w| *w < 0.0)
        || total <= 0.0
      {
        return Ok(unevaluated("RandomVariate", args));
      }
      let count = n.unwrap_or(1);
      let mut out = Vec::with_capacity(count);
      for _ in 0..count {
        let u: f64 = crate::with_rng(|rng| rng.random_range(0.0..total));
        let mut acc = 0.0;
        let mut idx = ws.len() - 1;
        for (i, w) in ws.iter().enumerate() {
          acc += w;
          if u < acc {
            idx = i;
            break;
          }
        }
        let draw = random_variate_ast(&[comps[idx].clone()])?;
        if matches!(&draw, Expr::FunctionCall { name, .. } if name == "RandomVariate")
        {
          return Ok(unevaluated("RandomVariate", args));
        }
        out.push(draw);
      }
      match n {
        None => Ok(out.remove(0)),
        Some(_) => Ok(Expr::List(out.into())),
      }
    }
    Expr::FunctionCall { name, args: dargs }
      if name == "ParetoDistribution" && dargs.len() == 2 =>
    {
      // Inverse-transform sampling: the CDF is 1 - (k/x)^alpha for x >= k,
      // so x = k * u^(-1/alpha) for u uniform on (0, 1].
      let (Some(k), Some(alpha)) =
        (try_eval_to_f64(&dargs[0]), try_eval_to_f64(&dargs[1]))
      else {
        return Ok(unevaluated("RandomVariate", args));
      };
      if k <= 0.0 || alpha <= 0.0 {
        return Ok(unevaluated("RandomVariate", args));
      }
      let sample = || -> Expr {
        let u: f64 =
          crate::with_rng(|rng| 1.0 - rng.random_range(0.0..1.0_f64));
        Expr::Real(k * u.powf(-1.0 / alpha))
      };
      match n {
        None => Ok(sample()),
        Some(count) => Ok(Expr::List(
          (0..count).map(|_| sample()).collect::<Vec<_>>().into(),
        )),
      }
    }
    Expr::FunctionCall { name, args: dargs }
      if name == "GeometricDistribution" && dargs.len() == 1 =>
    {
      // Number of failures before the first success: floor(ln u / ln(1 - p))
      // for u uniform on (0, 1].
      let Some(p) = try_eval_to_f64(&dargs[0]).filter(|p| *p > 0.0 && *p < 1.0)
      else {
        return Ok(unevaluated("RandomVariate", args));
      };
      let sample = || -> Expr {
        let u: f64 = 1.0 - crate::with_rng(|rng| rng.random_range(0.0..1.0));
        Expr::Integer((u.ln() / (-p).ln_1p()).floor() as i128)
      };
      match n {
        None => Ok(sample()),
        Some(count) => Ok(Expr::List(
          (0..count).map(|_| sample()).collect::<Vec<_>>().into(),
        )),
      }
    }
    Expr::FunctionCall { name, args: dargs }
      if name == "TruncatedDistribution" && dargs.len() == 2 =>
    {
      // Rejection sampling: draw from the base distribution until the value
      // falls inside {lo, hi}. A truncation keeping (almost) no mass would
      // never terminate, so give up after a bounded number of rejections.
      let Expr::List(bounds) = &dargs[0] else {
        return Ok(unevaluated("RandomVariate", args));
      };
      let bound = |e: &Expr| -> Option<f64> {
        match expr_to_string(e).as_str() {
          "Infinity" => Some(f64::INFINITY),
          "-Infinity" => Some(f64::NEG_INFINITY),
          _ => try_eval_to_f64(e),
        }
      };
      let (true, Some(lo), Some(hi)) = (
        bounds.len() == 2,
        bounds.first().and_then(bound),
        bounds.get(1).and_then(bound),
      ) else {
        return Ok(unevaluated("RandomVariate", args));
      };
      let draw = || -> Result<Option<Expr>, InterpreterError> {
        for _ in 0..100_000 {
          let candidate = random_variate_ast(&[dargs[1].clone()])?;
          match try_eval_to_f64(&candidate) {
            Some(x) if (lo..=hi).contains(&x) => return Ok(Some(candidate)),
            Some(_) => {}
            None => return Ok(None),
          }
        }
        Ok(None)
      };
      let mut out = Vec::with_capacity(n.unwrap_or(1));
      for _ in 0..n.unwrap_or(1) {
        match draw()? {
          Some(v) => out.push(v),
          None => return Ok(unevaluated("RandomVariate", args)),
        }
      }
      match n {
        None => Ok(out.remove(0)),
        Some(_) => Ok(Expr::List(out.into())),
      }
    }
    Expr::FunctionCall { name, args: dargs }
      if name == "PoissonDistribution" && dargs.len() == 1 =>
    {
      let lambda = try_eval_to_f64(&dargs[0]).ok_or_else(|| {
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
          let r = try_eval_to_f64(&dargs[0]).ok_or_else(|| {
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
                  (try_eval_to_f64(&items[0]), try_eval_to_f64(&items[1]))
              {
                Ok((a, b))
              } else {
                Err(InterpreterError::EvaluationError(format!(
                  "BinormalDistribution: invalid {label} (expected a 2-element list)"
                )))
              }
            };
          let (mu1, mu2) = extract_pair(&dargs[0], "mean vector")?;
          let (sigma1, sigma2) =
            extract_pair(&dargs[1], "standard deviations")?;
          let r = try_eval_to_f64(&dargs[2]).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "BinormalDistribution: invalid correlation".into(),
            )
          })?;
          (mu1, mu2, sigma1, sigma2, r)
        }
        _ => {
          return Ok(unevaluated("RandomVariate", args));
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
      if (name == "MultinormalDistribution" && dargs.len() == 2)
        || (name == "MultivariateTDistribution"
          && (dargs.len() == 2 || dargs.len() == 3)) =>
    {
      // MultivariateTDistribution[{mu,} sigma, nu] scales a centered
      // multinormal draw by Sqrt[nu / W] for a chi-square(nu) variate W.
      let student = name == "MultivariateTDistribution";
      let nu = if student {
        match try_eval_to_f64(dargs.last().unwrap()) {
          Some(v) if v > 0.0 => Some(v),
          _ => return Ok(unevaluated("RandomVariate", args)),
        }
      } else {
        None
      };
      let zero_mean;
      let (mu_arg, sigma_arg) = if student && dargs.len() == 2 {
        let Expr::List(rows) = &dargs[0] else {
          return Ok(unevaluated("RandomVariate", args));
        };
        zero_mean = Expr::List(vec![Expr::Integer(0); rows.len()].into());
        (&zero_mean, &dargs[0])
      } else {
        (&dargs[0], &dargs[1])
      };
      // MultinormalDistribution[mu, sigma]: a k-dimensional mean vector and
      // a k x k covariance matrix. Sampled via the standard reparametrization
      // `mu + L z`, where `z` is a vector of iid standard normals and `L` is
      // the lower-triangular Cholesky factor of `sigma` (`L L^T = sigma`),
      // computed numerically since (unlike PDF/CDF) a random draw has no
      // exact/symbolic form to preserve.
      let (Expr::List(mu_list), sigma_expr @ Expr::List(sigma_list)) =
        (mu_arg, sigma_arg)
      else {
        return Ok(unevaluated("RandomVariate", args));
      };
      let k = mu_list.len();
      if k == 0 || sigma_list.len() != k {
        return Ok(unevaluated("RandomVariate", args));
      }
      let mut mu = Vec::with_capacity(k);
      for m in mu_list {
        match try_eval_to_f64(m) {
          Some(v) => mu.push(v),
          None => return Ok(unevaluated("RandomVariate", args)),
        }
      }
      let mut sigma = vec![vec![0.0f64; k]; k];
      for (i, row) in sigma_list.iter().enumerate() {
        let Expr::List(cols) = row else {
          return Ok(unevaluated("RandomVariate", args));
        };
        if cols.len() != k {
          return Ok(unevaluated("RandomVariate", args));
        }
        for (j, c) in cols.iter().enumerate() {
          match try_eval_to_f64(c) {
            Some(v) => sigma[i][j] = v,
            None => return Ok(unevaluated("RandomVariate", args)),
          }
        }
      }

      let posdefprm = || {
        let position = if student && dargs.len() == 2 { 1 } else { 2 };
        crate::emit_message(&format!(
          "{name}::posdefprm: The value {} at position {position} in {} is expected to be a symmetric positive definite matrix.",
          expr_to_string(sigma_expr),
          expr_to_string(&unevaluated(name, dargs))
        ));
        Ok(unevaluated("RandomVariate", args))
      };

      let scale = sigma
        .iter()
        .flatten()
        .fold(1.0f64, |acc, v| acc.max(v.abs()));
      let tol = 1e-9 * scale;
      for i in 0..k {
        for j in 0..i {
          if (sigma[i][j] - sigma[j][i]).abs() > tol {
            return posdefprm();
          }
        }
      }

      // Cholesky-Banachiewicz decomposition; a non-positive pivot means
      // `sigma` is not positive definite, so the distribution does not exist.
      let mut l = vec![vec![0.0f64; k]; k];
      for i in 0..k {
        for j in 0..=i {
          let mut sum = sigma[i][j];
          for p in 0..j {
            sum -= l[i][p] * l[j][p];
          }
          if i == j {
            if sum <= tol.max(1e-12) {
              return posdefprm();
            }
            l[i][j] = sum.sqrt();
          } else {
            l[i][j] = sum / l[j][j];
          }
        }
      }

      let sample_vec = || -> Vec<f64> {
        use rand_distr::StandardNormal;
        let z: Vec<f64> = crate::with_rng(|rng| {
          (0..k).map(|_| rng.sample(StandardNormal)).collect()
        });
        let scale = match nu {
          Some(nu) => {
            let chi = rand_distr::ChiSquared::new(nu)
              .map_or(nu, |d| crate::with_rng(|rng| d.sample(rng)));
            (nu / chi).sqrt()
          }
          None => 1.0,
        };
        (0..k)
          .map(|i| mu[i] + scale * (0..=i).map(|j| l[i][j] * z[j]).sum::<f64>())
          .collect()
      };
      let vec_to_list =
        |v: Vec<f64>| Expr::List(v.into_iter().map(Expr::Real).collect());
      match n {
        None => Ok(vec_to_list(sample_vec())),
        Some(count) => {
          let mut out = Vec::with_capacity(count);
          for _ in 0..count {
            out.push(vec_to_list(sample_vec()));
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
      let mu0 = try_eval_to_f64(&dargs[0]).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "MultivariatePoissonDistribution: invalid shared rate".into(),
        )
      })?;
      let Expr::List(marginals_list) = &dargs[1] else {
        return Err(InterpreterError::EvaluationError(
          "MultivariatePoissonDistribution: expected a list of marginal rates"
            .into(),
        ));
      };
      let mut marginal_rates: Vec<f64> =
        Vec::with_capacity(marginals_list.len());
      for item in marginals_list {
        let mi = try_eval_to_f64(item).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "MultivariatePoissonDistribution: invalid marginal rate".into(),
          )
        })?;
        if mi < mu0 {
          return Err(InterpreterError::EvaluationError(format!(
            "MultivariatePoissonDistribution: each marginal must be >= mu0 ({mi} < {mu0})"
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
    Expr::FunctionCall { name, args: dargs }
      if name == "BinomialDistribution" && dargs.len() == 2 =>
    {
      // BinomialDistribution[n, p]: number of successes in `trials`
      // independent Bernoulli(p) trials.
      let trials = try_eval_to_f64(&dargs[0]).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "BinomialDistribution: invalid trial count".into(),
        )
      })?;
      let prob = try_eval_to_f64(&dargs[1]).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "BinomialDistribution: invalid success probability".into(),
        )
      })?;
      if trials < 0.0 || !(0.0..=1.0).contains(&prob) {
        return Ok(unevaluated("RandomVariate", args));
      }
      let trials = trials.round() as u64;
      let sample_binomial = || -> i128 {
        use rand_distr::{Binomial, Distribution};
        match Binomial::new(trials, prob) {
          Ok(b) => crate::with_rng(|rng| b.sample(rng) as i128),
          Err(_) => 0,
        }
      };
      match n {
        None => Ok(Expr::Integer(sample_binomial())),
        Some(count) => {
          let mut out = Vec::with_capacity(count);
          for _ in 0..count {
            out.push(Expr::Integer(sample_binomial()));
          }
          Ok(Expr::List(out.into()))
        }
      }
    }
    Expr::FunctionCall { name, args: dargs }
      if name == "BernoulliDistribution" && dargs.len() == 1 =>
    {
      // BernoulliDistribution[p]: 1 with probability p, 0 with probability
      // 1 - p (a single Binomial[1, p] trial).
      let prob = try_eval_to_f64(&dargs[0]).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "BernoulliDistribution: invalid success probability".into(),
        )
      })?;
      if !(0.0..=1.0).contains(&prob) {
        return Ok(unevaluated("RandomVariate", args));
      }
      let sample_bernoulli = || -> i128 {
        crate::with_rng(|rng| (rng.random_range(0.0..1.0) < prob) as i128)
      };
      match n {
        None => Ok(Expr::Integer(sample_bernoulli())),
        Some(count) => {
          let out: Vec<Expr> = (0..count)
            .map(|_| Expr::Integer(sample_bernoulli()))
            .collect();
          Ok(Expr::List(out.into()))
        }
      }
    }
    // CategoricalDistribution[{c1, …}, {p1, …}] draws category ci with
    // weight pi (uniformly when no weights are given).
    Expr::FunctionCall { name, args: dargs }
      if name == "CategoricalDistribution"
        && matches!(dargs.len(), 1 | 2)
        && matches!(&dargs[0], Expr::List(_)) =>
    {
      let spec = match dargs.get(1) {
        Some(w) => Expr::Rule {
          pattern: Box::new(w.clone()),
          replacement: Box::new(dargs[0].clone()),
        },
        None => dargs[0].clone(),
      };
      let mut choice_args = vec![spec];
      if let Some(count) = n {
        choice_args.push(Expr::Integer(count as i128));
      }
      let drawn = random_choice_ast(&choice_args)?;
      if matches!(&drawn, Expr::FunctionCall { name, .. } if name == "RandomChoice")
      {
        return Ok(unevaluated("RandomVariate", args));
      }
      Ok(drawn)
    }
    // Any other univariate distribution is sampled by inverse transform:
    // Quantile[dist, u] for u uniform on (0, 1). The first draw decides
    // whether Quantile knows the distribution at all.
    Expr::FunctionCall { .. } => {
      let draw = || -> Result<Option<Expr>, InterpreterError> {
        let u = crate::with_rng(|rng| {
          loop {
            let u: f64 = rng.random_range(0.0..1.0);
            if u > 0.0 {
              break u;
            }
          }
        });
        let q = crate::evaluator::evaluate_expr_to_expr(&call(
          "Quantile",
          vec![dist.clone(), Expr::Real(u)],
        ))?;
        Ok(match q {
          Expr::Integer(_) | Expr::BigInteger(_) => Some(q),
          Expr::Real(x) if x.is_finite() => Some(q),
          _ => try_eval_to_f64(&q)
            .filter(|x| x.is_finite())
            .map(Expr::Real),
        })
      };
      let Some(first) = draw()? else {
        return Ok(unevaluated("RandomVariate", args));
      };
      match n {
        None => Ok(first),
        Some(count) => {
          let mut out = Vec::with_capacity(count);
          out.push(first);
          for _ in 1..count {
            match draw()? {
              Some(x) => out.push(x),
              None => return Ok(unevaluated("RandomVariate", args)),
            }
          }
          Ok(Expr::List(out.into()))
        }
      }
    }
    _ => Ok(unevaluated("RandomVariate", args)),
  }
}

/// RandomPrime[imax] or RandomPrime[{imin, imax}] or RandomPrime[range, n]
pub fn random_prime_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use rand::RngExt;

  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "RandomPrime expects 1 or 2 arguments".into(),
    ));
  }

  // wolframscript never raises a hard error here: a bound that is not a
  // positive integer emits `RandomPrime::intp` and an otherwise-valid interval
  // that contains no primes emits `RandomPrime::noprime`. In both cases it
  // returns the expression unevaluated.
  let uneval = || Ok(unevaluated("RandomPrime", args));

  // Parse the range from first argument
  let (min, max) = match &args[0] {
    Expr::Integer(imax) => {
      if *imax < 1 {
        crate::emit_message(&format!(
          "RandomPrime::intp: {imax} is not a positive integer."
        ));
        return uneval();
      }
      (2i128, *imax)
    }
    Expr::BigInteger(imax) if imax.sign() == num_bigint::Sign::Plus => {
      return Ok(random_big_prime(&num_bigint::BigInt::from(2), imax, args));
    }
    Expr::Real(_) => {
      crate::emit_message(&format!(
        "RandomPrime::intp: {} is not a positive integer.",
        expr_to_string(&args[0])
      ));
      return uneval();
    }
    Expr::List(items) if items.len() == 2 => {
      // Both bounds must be positive integers; report the first offender.
      for it in items {
        match it {
          Expr::Integer(b) if *b >= 1 => {}
          Expr::BigInteger(b) if b.sign() == num_bigint::Sign::Plus => {}
          Expr::Integer(_) | Expr::Real(_) => {
            crate::emit_message(&format!(
              "RandomPrime::intp: {} is not a positive integer.",
              expr_to_string(it)
            ));
            return uneval();
          }
          // A symbolic bound stays unevaluated (no message), matching WL.
          _ => return uneval(),
        }
      }
      if let (Expr::Integer(a), Expr::Integer(b)) = (&items[0], &items[1]) {
        // wolframscript orders the bounds, so {5, 2} samples from [2, 5].
        ((*a).min(*b), (*a).max(*b))
      } else {
        // Bounds past i128 sample from the BigInteger range.
        let big = |e: &Expr| crate::functions::math_ast::expr_to_bigint(e);
        let (Some(a), Some(b)) = (big(&items[0]), big(&items[1])) else {
          return uneval();
        };
        return Ok(random_big_prime(
          &a.clone().min(b.clone()),
          &a.max(b),
          args,
        ));
      }
    }
    _ => return uneval(),
  };

  // The second arg may be a positive integer (length of a flat list)
  // or a list of dimensions {n1, n2, ...} for a nested array. Track
  // both as a `dims` vector — when it has length 1 we keep the
  // original flat-list output to match wolframscript.
  // The dimensions parameter must be a positive integer or a list of positive
  // integers; anything else emits posdim and stays unevaluated (rather than a
  // hard evaluation error).
  let posdim = || {
    crate::emit_message(&format!(
      "RandomPrime::posdim: The dimensions parameter {} is expected to be a positive integer or a list of positive integers.",
      expr_to_string(&args[1])
    ));
    Ok(unevaluated("RandomPrime", args))
  };
  let dims: Vec<usize> = if args.len() == 2 {
    match &args[1] {
      Expr::Integer(n) if *n > 0 => vec![*n as usize],
      Expr::List(items) if !items.is_empty() => {
        let mut ds = Vec::with_capacity(items.len());
        for it in items {
          match it {
            Expr::Integer(n) if *n > 0 => ds.push(*n as usize),
            _ => return posdim(),
          }
        }
        ds
      }
      _ => return posdim(),
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
      crate::emit_message(
        "RandomPrime::noprime: There are no primes in the specified interval.",
      );
      return uneval();
    }
    if dims.is_empty() {
      let idx = crate::with_rng(|rng| rng.random_range(0..primes.len()));
      Ok(Expr::Integer(primes[idx]))
    } else {
      let flat: Vec<Expr> = crate::with_rng(|rng| {
        (0..count)
          .map(|_| Expr::Integer(primes[rng.random_range(0..primes.len())]))
          .collect()
      });
      Ok(reshape_into_dims(flat, &dims))
    }
  } else {
    // Rejection sampling for large ranges
    let start = if min < 2 { 2i128 } else { min };
    if start > max {
      crate::emit_message(
        "RandomPrime::noprime: There are no primes in the specified interval.",
      );
      return uneval();
    }
    let mut results = Vec::with_capacity(count);
    for _ in 0..count {
      let prime = crate::with_rng(|rng| {
        loop {
          let candidate = rng.random_range(start..=max);
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

/// `RandomPrime[{min, max}, dims]` for BigInteger bounds: uniform candidates
/// in the range, kept when prime. The interval is assumed to hold primes,
/// which every range this wide does.
fn random_big_prime(
  min: &num_bigint::BigInt,
  max: &num_bigint::BigInt,
  args: &[Expr],
) -> Expr {
  use num_bigint::BigInt;
  use rand::RngExt;
  let span = max - min + BigInt::from(1);
  let bits = span.bits();
  let draw = || -> BigInt {
    crate::with_rng(|rng| {
      loop {
        let words = bits.div_ceil(64) as usize;
        let mut digits: Vec<u64> = (0..words).map(|_| rng.random()).collect();
        let extra = words as u64 * 64 - bits;
        if extra > 0
          && let Some(top) = digits.last_mut()
        {
          *top >>= extra;
        }
        let v = BigInt::from(num_bigint::BigUint::from_slice(
          &digits
            .iter()
            .flat_map(|d| [*d as u32, (*d >> 32) as u32])
            .collect::<Vec<u32>>(),
        ));
        if v < span {
          return min + v;
        }
      }
    })
  };
  let prime = || loop {
    let candidate = draw();
    if crate::functions::math_ast::is_prime_bigint(&candidate) {
      return crate::functions::math_ast::bigint_to_expr(candidate);
    }
  };
  let dims: Vec<i128> = match args.get(1) {
    None => return prime(),
    Some(Expr::Integer(n)) if *n > 0 => vec![*n],
    Some(Expr::List(ds))
      if !ds.is_empty()
        && ds.iter().all(|d| matches!(d, Expr::Integer(n) if *n > 0)) =>
    {
      ds.iter()
        .map(|d| if let Expr::Integer(n) = d { *n } else { 0 })
        .collect()
    }
    _ => return unevaluated("RandomPrime", args),
  };
  fn nested(dims: &[i128], prime: &dyn Fn() -> Expr) -> Expr {
    match dims.split_first() {
      None => prime(),
      Some((n, rest)) => Expr::List(
        (0..*n)
          .map(|_| nested(rest, prime))
          .collect::<Vec<_>>()
          .into(),
      ),
    }
  }
  nested(&dims, &prime)
}

/// Collect all primes in [min, max] (for small ranges)
fn collect_primes_in_range(min: i128, max: i128) -> Vec<i128> {
  let start = if min < 2 { 2 } else { min };
  let mut primes = Vec::new();
  for n in start..=max {
    if is_prime_i128(n) {
      primes.push(n);
    }
  }
  primes
}

/// SeedRandom[n] - Seed the random number generator
/// SeedRandom[] - Reset to non-deterministic RNG
pub fn seed_random_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match args.len() {
    0 => {
      crate::unseed_rng();
      Ok(null_expr())
    }
    1 => match &args[0] {
      Expr::Integer(seed) => {
        crate::seed_rng(*seed as u64);
        Ok(null_expr())
      }
      Expr::String(s) => {
        // Wolfram accepts string seeds; hash the string deterministically
        // so the same string always produces the same sequence.
        crate::seed_rng(hash_string_to_u64(s));
        Ok(null_expr())
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

/// Edges `{1≤i<j≤n}` of `m` distinct pairs drawn uniformly at random.
fn uniform_random_edges(n: usize, m: usize) -> Vec<(usize, usize)> {
  use rand::seq::SliceRandom;
  let mut all_edges: Vec<(usize, usize)> = Vec::new();
  for i in 1..=n {
    for j in (i + 1)..=n {
      all_edges.push((i, j));
    }
  }
  crate::with_rng(|rng| all_edges.shuffle(rng));
  all_edges.truncate(m);
  all_edges.sort_unstable();
  all_edges
}

/// Barabási–Albert preferential attachment: start from the complete graph
/// on `k + 1` vertices, then attach every further vertex to `k` distinct
/// existing vertices chosen with probability proportional to their degree.
fn barabasi_albert_edges(n: usize, k: usize) -> Vec<(usize, usize)> {
  use rand::RngExt;
  let mut edges: Vec<(usize, usize)> = Vec::new();
  let seed = (k + 1).min(n);
  for i in 1..=seed {
    for j in (i + 1)..=seed {
      edges.push((i, j));
    }
  }
  // Every edge endpoint appears once per incident edge, so a uniform pick
  // from this list is a degree-proportional pick of a vertex.
  let mut endpoints: Vec<usize> =
    edges.iter().flat_map(|&(a, b)| [a, b]).collect();
  crate::with_rng(|rng| {
    for v in (seed + 1)..=n {
      let mut targets: Vec<usize> = Vec::with_capacity(k);
      while targets.len() < k.min(v - 1) {
        let t = if endpoints.is_empty() {
          rng.random_range(1..v)
        } else {
          endpoints[rng.random_range(0..endpoints.len())]
        };
        if !targets.contains(&t) {
          targets.push(t);
        }
      }
      targets.sort_unstable();
      for t in targets {
        edges.push((t, v));
        endpoints.push(t);
        endpoints.push(v);
      }
    }
  });
  edges
}

/// Watts–Strogatz small world: the circulant ring where every vertex is
/// joined to its `k` nearest neighbours on each side, with every edge's far
/// endpoint rewired with probability `p` to a uniformly chosen vertex that
/// keeps the graph simple.
fn watts_strogatz_edges(n: usize, p: f64, k: usize) -> Vec<(usize, usize)> {
  use rand::RngExt;
  use std::collections::HashSet;
  let key = |a: usize, b: usize| (a.min(b), a.max(b));
  let mut edges: Vec<(usize, usize)> = Vec::new();
  let mut present: HashSet<(usize, usize)> = HashSet::new();
  for d in 1..=k {
    for i in 1..=n {
      let j = (i - 1 + d) % n + 1;
      if i != j && present.insert(key(i, j)) {
        edges.push((i, j));
      }
    }
  }
  crate::with_rng(|rng| {
    for e in &mut edges {
      if !rng.random_bool(p) {
        continue;
      }
      let (u, v) = *e;
      let free: Vec<usize> = (1..=n)
        .filter(|&w| w != u && !present.contains(&key(u, w)))
        .collect();
      if free.is_empty() {
        continue;
      }
      let w = free[rng.random_range(0..free.len())];
      present.remove(&key(u, v));
      present.insert(key(u, w));
      *e = (u, w);
    }
  });
  edges
}

/// Erdős–Gallai: whether a degree sequence is that of a simple graph.
fn is_graphical(degrees: &[usize]) -> bool {
  let mut d = degrees.to_vec();
  d.sort_unstable_by(|a, b| b.cmp(a));
  if d.iter().sum::<usize>() % 2 == 1 {
    return false;
  }
  let n = d.len();
  for k in 1..=n {
    let left: usize = d[..k].iter().sum();
    let right: usize =
      k * (k - 1) + d[k..].iter().map(|&x| x.min(k)).sum::<usize>();
    if left > right {
      return false;
    }
  }
  true
}

/// A simple graph with the given (graphical) degree sequence: Havel–Hakimi,
/// then random double-edge swaps that keep every degree and the graph simple.
fn degree_sequence_edges(degrees: &[usize]) -> Vec<(usize, usize)> {
  use rand::RngExt;
  let n = degrees.len();
  let mut remaining: Vec<(usize, usize)> = degrees
    .iter()
    .enumerate()
    .map(|(v, &d)| (d, v + 1))
    .collect();
  let mut edges: Vec<(usize, usize)> = Vec::new();
  loop {
    remaining.sort_unstable_by(|a, b| b.cmp(a));
    let Some(&(d, v)) = remaining.first() else {
      break;
    };
    if d == 0 {
      break;
    }
    for item in remaining.iter_mut().skip(1).take(d) {
      item.0 -= 1;
      edges.push((v.min(item.1), v.max(item.1)));
    }
    remaining[0].0 = 0;
  }
  let key = |a: usize, b: usize| (a.min(b), a.max(b));
  let mut present: std::collections::HashSet<(usize, usize)> =
    edges.iter().copied().collect();
  let m = edges.len();
  if m >= 2 && n >= 4 {
    crate::with_rng(|rng| {
      for _ in 0..10 * m {
        let i = rng.random_range(0..m);
        let j = rng.random_range(0..m);
        if i == j {
          continue;
        }
        let ((a, b), (c, d)) = (edges[i], edges[j]);
        let (a, b) = if rng.random_bool(0.5) { (a, b) } else { (b, a) };
        // a-b, c-d  →  a-c, b-d
        if a == c || a == d || b == c || b == d {
          continue;
        }
        let (e1, e2) = (key(a, c), key(b, d));
        if present.contains(&e1) || present.contains(&e2) {
          continue;
        }
        present.remove(&edges[i]);
        present.remove(&edges[j]);
        present.insert(e1);
        present.insert(e2);
        edges[i] = e1;
        edges[j] = e2;
      }
    });
  }
  edges
}

/// RandomGraph[{n, m}] — undirected graph with n vertices and m random edges.
/// RandomGraph[gdist] — a graph drawn from a random graph distribution.
/// RandomGraph[spec, k] — list of k such graphs.
pub fn random_graph_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use rand::RngExt;

  let unevaluated = || unevaluated("RandomGraph", args);

  // Split positional args from trailing Rule options.
  let mut positional: Vec<&Expr> = Vec::new();
  let mut options: Vec<Expr> = Vec::new();
  for a in args {
    if matches!(a, Expr::Rule { .. } | Expr::RuleDelayed { .. }) {
      options.push(a.clone());
    } else {
      positional.push(a);
    }
  }

  if positional.is_empty() || positional.len() > 2 {
    return Ok(unevaluated());
  }

  let count = |e: &Expr| match expr_to_i128(e) {
    Some(n) if n >= 0 => Some(n as usize),
    _ => None,
  };
  let probability = |e: &Expr| match try_eval_to_f64(e) {
    Some(p) if (0.0..=1.0).contains(&p) => Some(p),
    _ => None,
  };

  // Each spec becomes a vertex count plus a sampler for one edge set.
  type Sampler = Box<dyn Fn() -> Vec<(usize, usize)>>;
  // SpatialGraphDistribution[n, r]: n uniform points in the unit square,
  // joined when they are within r; the points are the vertex coordinates.
  let spatial = match positional[0] {
    Expr::FunctionCall { name, args: d }
      if name == "SpatialGraphDistribution" && d.len() >= 2 =>
    {
      let (Some(n), Some(r)) = (count(&d[0]), try_eval_to_f64(&d[1])) else {
        return Ok(unevaluated());
      };
      if r <= 0.0 {
        return Ok(unevaluated());
      }
      Some((n, r))
    }
    _ => None,
  };
  if let Some((n, r)) = spatial {
    let k = match positional.get(1) {
      None => None,
      Some(e) => match count(e) {
        Some(k) => Some(k),
        None => return Ok(unevaluated()),
      },
    };
    let one = || {
      let points: Vec<(f64, f64)> = crate::with_rng(|rng| {
        (0..n)
          .map(|_| (rng.random::<f64>(), rng.random::<f64>()))
          .collect()
      });
      let mut edges = Vec::new();
      for i in 0..n {
        for j in (i + 1)..n {
          let (dx, dy) = (points[i].0 - points[j].0, points[i].1 - points[j].1);
          if dx.hypot(dy) <= r {
            edges.push(call(
              "UndirectedEdge",
              vec![Expr::Integer(i as i128 + 1), Expr::Integer(j as i128 + 1)],
            ));
          }
        }
      }
      let coords: Vec<Expr> = points
        .iter()
        .map(|&(x, y)| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into()))
        .collect();
      let mut opts = vec![Expr::Rule {
        pattern: Box::new(Expr::Identifier("VertexCoordinates".into())),
        replacement: Box::new(Expr::List(coords.into())),
      }];
      opts.extend(options.iter().cloned());
      call(
        "Graph",
        vec![
          Expr::List(
            (1..=n)
              .map(|v| Expr::Integer(v as i128))
              .collect::<Vec<_>>()
              .into(),
          ),
          Expr::List(edges.into()),
          Expr::List(opts.into()),
        ],
      )
    };
    return Ok(match k {
      None => one(),
      Some(k) => Expr::List((0..k).map(|_| one()).collect::<Vec<_>>().into()),
    });
  }
  let (n, sample): (usize, Sampler) = match positional[0] {
    Expr::List(items) if items.len() == 2 => {
      let (Some(n), Some(m)) = (count(&items[0]), count(&items[1])) else {
        return Ok(unevaluated());
      };
      if m > n.saturating_mul(n.saturating_sub(1)) / 2 {
        return Ok(unevaluated());
      }
      (n, Box::new(move || uniform_random_edges(n, m)))
    }
    Expr::FunctionCall { name, args: d } => match (name.as_str(), d.len()) {
      ("UniformGraphDistribution", 2) => {
        let (Some(n), Some(m)) = (count(&d[0]), count(&d[1])) else {
          return Ok(unevaluated());
        };
        if m > n.saturating_mul(n.saturating_sub(1)) / 2 {
          return Ok(unevaluated());
        }
        (n, Box::new(move || uniform_random_edges(n, m)))
      }
      // Erdős–Rényi G(n, p): each potential edge is independently
      // included with probability p.
      ("BernoulliGraphDistribution", 2) => {
        let (Some(n), Some(p)) = (count(&d[0]), probability(&d[1])) else {
          return Ok(unevaluated());
        };
        let sampler = move || {
          let mut edges = Vec::new();
          crate::with_rng(|rng| {
            for i in 1..=n {
              for j in (i + 1)..=n {
                if rng.random_bool(p) {
                  edges.push((i, j));
                }
              }
            }
          });
          edges
        };
        (n, Box::new(sampler))
      }
      ("BarabasiAlbertGraphDistribution", 2) => {
        let (Some(n), Some(k)) = (count(&d[0]), count(&d[1])) else {
          return Ok(unevaluated());
        };
        if k == 0 {
          return Ok(unevaluated());
        }
        (n, Box::new(move || barabasi_albert_edges(n, k)))
      }
      // A random simple graph with the given degree sequence: a
      // Havel–Hakimi realization shuffled by degree-preserving edge swaps.
      ("DegreeGraphDistribution", 1) => {
        let degrees: Option<Vec<usize>> = match &d[0] {
          Expr::List(ds) => ds.iter().map(count).collect(),
          _ => None,
        };
        let Some(degrees) = degrees.filter(|ds| is_graphical(ds)) else {
          crate::emit_message(&format!(
            "DegreeGraphDistribution::args: {} called with invalid parameters.",
            crate::syntax::expr_to_string(&unevaluated())
          ));
          return Ok(unevaluated());
        };
        (
          degrees.len(),
          Box::new(move || degree_sequence_edges(&degrees)),
        )
      }
      ("WattsStrogatzGraphDistribution", 2 | 3) => {
        let k = if d.len() == 3 { count(&d[2]) } else { Some(2) };
        let (Some(n), Some(p), Some(k)) = (count(&d[0]), probability(&d[1]), k)
        else {
          return Ok(unevaluated());
        };
        if k == 0 || 2 * k >= n {
          return Ok(unevaluated());
        }
        (n, Box::new(move || watts_strogatz_edges(n, p, k)))
      }
      _ => return Ok(unevaluated()),
    },
    _ => return Ok(unevaluated()),
  };

  let k = match positional.get(1) {
    None => None,
    Some(e) => match count(e) {
      Some(k) => Some(k),
      None => return Ok(unevaluated()),
    },
  };

  let make_graph = |edges: Vec<(usize, usize)>| -> Expr {
    let vertex_list: Vec<Expr> =
      (1..=n).map(|v| Expr::Integer(v as i128)).collect();
    let edge_list: Vec<Expr> = edges
      .into_iter()
      .map(|(a, b)| {
        call(
          "UndirectedEdge",
          vec![Expr::Integer(a as i128), Expr::Integer(b as i128)],
        )
      })
      .collect();
    let mut graph_args =
      vec![Expr::List(vertex_list.into()), Expr::List(edge_list.into())];
    if !options.is_empty() {
      graph_args.push(Expr::List(options.clone().into()));
    }
    call("Graph", graph_args)
  };

  match k {
    None => Ok(make_graph(sample())),
    Some(k) => {
      let graphs: Vec<Expr> = (0..k).map(|_| make_graph(sample())).collect();
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
