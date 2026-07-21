#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator, unevaluated};
use num_bigint::BigInt;

/// Pochhammer[a, n] - Rising factorial (Pochhammer symbol): a * (a+1) * ... * (a+n-1)
pub fn pochhammer_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Pochhammer expects exactly 2 arguments".into(),
    ));
  }
  // Pochhammer[a, 0] = 1 for any a
  if matches!(&args[1], Expr::Integer(0)) {
    return Ok(Expr::Integer(1));
  }
  // Both arguments are numeric integers → compute directly
  if let (Some(a), Some(n)) = (expr_to_i128(&args[0]), expr_to_i128(&args[1])) {
    if n < 0 {
      // Pochhammer[a, -n] = 1/((a-1)(a-2)...(a-n))
      let abs_n = (-n) as usize;
      let mut denom = BigInt::from(1);
      for i in 1..=abs_n as i128 {
        denom *= BigInt::from(a - i);
      }
      let denom_expr = bigint_to_expr(denom);
      let result = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(denom_expr),
      };
      return crate::evaluator::evaluate_expr_to_expr(&result);
    }
    let mut result = BigInt::from(1);
    for i in 0..n {
      result *= BigInt::from(a + i);
    }
    Ok(bigint_to_expr(result))
  } else if let Some(n) = expr_to_i128(&args[1]) {
    // n is a concrete integer, a is symbolic → expand symbolically
    let a_expr = &args[0];
    // Exact rational base p/q with a concrete integer n: compute the rising
    // (or falling) factorial as an exact BigInt rational, with no degree cap.
    // Pochhammer[p/q, n] = ∏_{i=0}^{n-1} (p + i q) / q^n. The symbolic path
    // below caps at |n| <= 20, which left e.g. Pochhammer[1/2, 30] unevaluated.
    if let (Expr::FunctionCall { name, args: ra }, _) = (a_expr, ())
      && name == "Rational"
      && ra.len() == 2
      && let (Expr::Integer(p), Expr::Integer(q)) = (&ra[0], &ra[1])
    {
      let p = BigInt::from(*p);
      let q = BigInt::from(*q);
      if n > 0 {
        let mut num = BigInt::from(1);
        let mut den = BigInt::from(1);
        for i in 0..n {
          num *= &p + BigInt::from(i) * &q;
          den *= &q;
        }
        return Ok(make_rational_expr(num, den));
      } else {
        // Pochhammer[a, -k] = 1/∏_{i=1}^{k} (a - i); a - i = (p - i q)/q.
        let abs_n = -n;
        let mut num = BigInt::from(1);
        let mut den = BigInt::from(1);
        for i in 1..=abs_n {
          num *= &p - BigInt::from(i) * &q;
          den *= &q;
        }
        // result = 1 / (num/den) = den/num
        return Ok(make_rational_expr(den, num));
      }
    }
    if n > 0 {
      // Pochhammer[a, n] = a * (a+1) * ... * (a+n-1). Expanded for any concrete
      // positive n (wolframscript expands the full product, with no small cap).
      let factors: Vec<Expr> = (0..n)
        .map(|i| {
          if i == 0 {
            a_expr.clone()
          } else {
            Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(Expr::Integer(i)),
              right: Box::new(a_expr.clone()),
            }
          }
        })
        .collect();
      let product = factors
        .into_iter()
        .reduce(|acc, f| Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(acc),
          right: Box::new(f),
        })
        .unwrap();
      crate::evaluator::evaluate_expr_to_expr(&product)
    } else if n < 0 {
      // Pochhammer[a, -n] = 1/((a-1)(a-2)...(a-n))
      let abs_n = (-n) as usize;
      let factors: Vec<Expr> = (1..=abs_n)
        .map(|i| Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::Integer(-(i as i128))),
          right: Box::new(a_expr.clone()),
        })
        .collect();
      let denom = factors
        .into_iter()
        .reduce(|acc, f| Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(acc),
          right: Box::new(f),
        })
        .unwrap();
      let result = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(denom),
      };
      crate::evaluator::evaluate_expr_to_expr(&result)
    } else {
      Ok(unevaluated("Pochhammer", args))
    }
  } else {
    // Exact numeric a with a non-integer rational b (integer b is handled
    // above): Pochhammer[a, b] = Gamma[a + b]/Gamma[a]. wolframscript reduces
    // these to closed form (e.g. Pochhammer[2, 1/2] -> (3 Sqrt[Pi])/4,
    // Pochhammer[2, -1/2] -> Sqrt[Pi]/2). Woxi's Gamma already performs the
    // same half-integer reduction, so evaluate the ratio symbolically.
    let is_exact_number = |e: &Expr| {
      matches!(e, Expr::Integer(_))
        || matches!(e, Expr::FunctionCall { name, args }
          if name == "Rational" && args.len() == 2)
    };
    if is_exact_number(&args[0]) && is_exact_number(&args[1]) {
      let gamma = |arg: Expr| Expr::FunctionCall {
        name: "Gamma".to_string(),
        args: vec![arg].into(),
      };
      let sum = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(args[0].clone()),
        right: Box::new(args[1].clone()),
      };
      let ratio = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(gamma(sum)),
        right: Box::new(gamma(args[0].clone())),
      };
      return crate::evaluator::evaluate_expr_to_expr(&ratio);
    }
    // Numeric evaluation: Pochhammer[a, n] = Gamma[a + n] / Gamma[a]
    if let (Some(a_f), Some(n_f)) =
      (try_eval_to_f64(&args[0]), try_eval_to_f64(&args[1]))
    {
      let has_real =
        matches!(&args[0], Expr::Real(_)) || matches!(&args[1], Expr::Real(_));
      if has_real {
        let gamma_a = super::gamma_fn(a_f);
        let gamma_a_n = super::gamma_fn(a_f + n_f);
        if gamma_a.is_finite()
          && gamma_a_n.is_finite()
          && gamma_a.abs() > 1e-300
        {
          return Ok(Expr::Real(gamma_a_n / gamma_a));
        }
      }
    }
    Ok(unevaluated("Pochhammer", args))
  }
}

/// FactorialPower[n, k] - falling factorial: n*(n-1)*...*(n-k+1)
/// FactorialPower[n, k, h] - generalized: n*(n-h)*(n-2h)*... (k terms)
/// Negative order: FactorialPower[n, -k, h] = 1/((n+h)*(n+2h)*...*(n+kh))
pub fn factorial_power_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "FactorialPower expects 2 or 3 arguments".into(),
    ));
  }
  let unevaluated = |args: &[Expr]| unevaluated("FactorialPower", args);

  let Some(k) = expr_to_i128(&args[1]) else {
    return Ok(unevaluated(args));
  };
  let h_expr = args.get(2).cloned().unwrap_or(Expr::Integer(1));

  // All-integer fast path with BigInt products (no i128 overflow)
  if let (Some(n), Some(h)) = (expr_to_i128(&args[0]), expr_to_i128(&h_expr)) {
    if k == 0 {
      return Ok(Expr::Integer(1));
    }
    let mut result = BigInt::from(1);
    if k > 0 {
      for i in 0..k {
        result *= BigInt::from(n - i * h);
      }
      return Ok(bigint_to_expr(result));
    }
    // k < 0: reciprocal of (n+h)*(n+2h)*...*(n+|k|h)
    for i in 1..=(-k) {
      result *= BigInt::from(n + i * h);
    }
    if result == BigInt::from(0) {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    if let Ok(den) = i128::try_from(result) {
      return Ok(make_rational(1, den));
    }
    return Ok(unevaluated(args));
  }

  // General numeric path (Real or Rational n/h): build the product as
  // expression arithmetic so exact rationals stay exact and reals fold
  // to machine precision.
  if expr_to_num(&args[0]).is_some() && expr_to_num(&h_expr).is_some() {
    if k == 0 {
      return Ok(Expr::Integer(1));
    }
    let term = |i: i128| Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        args[0].clone(),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(i), h_expr.clone()].into(),
        },
      ]
      .into(),
    };
    let factors: Vec<Expr> = if k > 0 {
      (0..k).map(|i| term(-i)).collect()
    } else {
      (1..=(-k)).map(term).collect()
    };
    let product =
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Times".to_string(),
        args: factors.into(),
      })?;
    if k > 0 {
      return Ok(product);
    }
    if matches!(&product, Expr::Integer(0)) {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![product, Expr::Integer(-1)].into(),
    });
  }

  Ok(unevaluated(args))
}

/// Gamma[n] - Gamma function: Gamma[n] = (n-1)! for positive integers
pub fn gamma_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Gamma expects 1, 2, or 3 arguments".into(),
    ));
  }

  // Three-argument generalized incomplete gamma:
  //   Gamma[a, z0, z1] = Gamma[a, z0] - Gamma[a, z1]
  // Wolfram keeps this symbolic; only a few special cases reduce.
  if args.len() == 3 {
    let (a, z0, z1) = (&args[0], &args[1], &args[2]);
    // Gamma[a, z, z] = 0
    if crate::syntax::expr_to_string(z0) == crate::syntax::expr_to_string(z1) {
      return Ok(Expr::Integer(0));
    }
    // Gamma[a, z0, Infinity] = Gamma[a, z0]   (Gamma[a, Infinity] = 0)
    if matches!(z1, Expr::Identifier(s) if s == "Infinity") {
      return gamma_incomplete_upper(a, z0);
    }
    // Gamma[1, z0, z1] = E^(-z0) - E^(-z1) (since Gamma[1, z] = E^(-z)).
    // wolframscript expands this for a = 1 regardless of the limits, unlike the
    // general finite-limit case which stays symbolic.
    if matches!(a, Expr::Integer(1)) {
      let g0 = gamma_incomplete_upper(a, z0)?;
      let g1 = gamma_incomplete_upper(a, z1)?;
      return crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(g0),
        right: Box::new(g1),
      });
    }
    // Numeric when any argument is an inexact (machine) real. Otherwise
    // Wolfram keeps the symbolic Gamma[a, z0, z1] form (it does NOT expand
    // to the difference of one-argument incomplete gammas).
    let has_real = matches!(a, Expr::Real(_))
      || matches!(z0, Expr::Real(_))
      || matches!(z1, Expr::Real(_));
    if has_real {
      let g0 = gamma_incomplete_upper(a, z0)?;
      let g1 = gamma_incomplete_upper(a, z1)?;
      return crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(g0),
        right: Box::new(g1),
      });
    }
    return Ok(unevaluated("Gamma", args));
  }

  // Two-argument form: Gamma[a, z] = upper incomplete gamma function
  if args.len() == 2 {
    return gamma_incomplete_upper(&args[0], &args[1]);
  }

  // Limits at infinity: Gamma[Infinity] = Infinity; the undirected
  // Gamma[ComplexInfinity] = ComplexInfinity; Gamma[-Infinity] is
  // Indeterminate (the poles at the negative integers accumulate there).
  match &args[0] {
    Expr::Identifier(s) if s == "Infinity" => {
      return Ok(Expr::Identifier("Infinity".to_string()));
    }
    Expr::Identifier(s) if s == "ComplexInfinity" => {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } if matches!(operand.as_ref(), Expr::Identifier(s) if s == "Infinity") => {
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    _ => {}
  }

  match expr_to_i128(&args[0]) {
    Some(n) => {
      if n <= 0 {
        // Gamma has poles at non-positive integers
        return Ok(Expr::Identifier("ComplexInfinity".to_string()));
      }
      // Gamma[n] = (n-1)! for positive integers
      let mut result = BigInt::from(1);
      for i in 2..n {
        result *= i;
      }
      Ok(bigint_to_expr(result))
    }
    None if matches!(&args[0], Expr::Real(_)) => {
      let f = if let Expr::Real(f) = &args[0] {
        *f
      } else {
        unreachable!()
      };
      if f <= 0.0 && f.fract() == 0.0 {
        // Poles at non-positive integers
        return Ok(Expr::Identifier("ComplexInfinity".to_string()));
      }
      // An integer-valued real gives the exact factorial Gamma[n] = (n-1)!,
      // rounded to a machine real: Gamma[5.0] -> 24., not the float-Lanczos
      // 23.999.... 170! is the largest below the f64 range. Beyond that the
      // numeric path below handles the overflow.
      if f.fract() == 0.0 && (1.0..=171.0).contains(&f) {
        use num_traits::ToPrimitive;
        let n = f as i128; // Gamma[n] = (n-1)!
        let mut result = BigInt::from(1);
        for i in 2..n {
          result *= i;
        }
        return Ok(Expr::Real(result.to_f64().unwrap_or(f64::INFINITY)));
      }
      // For arguments beyond Wolfram's machine-precision overflow band
      // (around 1e14), wolframscript emits General::ovfl and returns
      // Overflow[]. Match that behavior — f64 can't represent values
      // out here anyway.
      if f >= 1.0e14 {
        crate::emit_message("General::ovfl: Overflow occurred in computation.");
        return Ok(Expr::FunctionCall {
          name: "Overflow".to_string(),
          args: vec![].into(),
        });
      }
      // Use Stirling's approximation via the standard library's tgamma equivalent
      // Rust doesn't have tgamma in std, but we can compute via the Lanczos approximation
      let result = gamma_fn(f);
      if result.is_infinite() {
        Ok(Expr::Identifier("ComplexInfinity".to_string()))
      } else {
        Ok(Expr::Real(result))
      }
    }
    // Gamma[n/2] for odd n — half-integer values
    // Gamma[1/2] = Sqrt[Pi], Gamma[3/2] = Sqrt[Pi]/2,
    // Gamma[(2k+1)/2] = (2k)! * Sqrt[Pi] / (4^k * k!)
    // Gamma[(-2k+1)/2] uses reflection: Gamma[-n+1/2] = (-1)^n * Pi / (Gamma[n+1/2] * Sin(Pi*(n+1/2)))
    //   simplified: Gamma[1/2 - n] = (-4)^n * n! * Sqrt[Pi] / (2n)!
    _ if matches!(&args[0], Expr::FunctionCall { name, args: ra }
      if name == "Rational" && ra.len() == 2
        && matches!(&ra[1], Expr::Integer(2))
        && matches!(&ra[0], Expr::Integer(n) if n % 2 != 0)
    ) =>
    {
      if let Expr::FunctionCall { args: ra, .. } = &args[0]
        && let Expr::Integer(num) = &ra[0]
      {
        let num = *num;
        // num is odd, denominator is 2, so argument is num/2
        if num > 0 {
          // Positive half-integers: Gamma[(1+2n)/2] where n = (num-1)/2
          // Gamma[(2n+1)/2] = (2n)! * Sqrt[Pi] / (4^n * n!)
          let n = (num - 1) / 2;
          let mut numer = BigInt::from(1);
          for i in (n + 1)..=(2 * n) {
            numer *= i;
          }
          let denom = BigInt::from(4).pow(n as u32);
          return Ok(gamma_half_expr(numer, denom, false));
        } else if num < 0 {
          // Negative half-integers: Gamma[(1-2n)/2] where n = (1-num)/2
          // Gamma[1/2 - n] = (-4)^n * n! * Sqrt[Pi] / (2n)!
          let n = (1 - num) / 2;
          let is_neg = n % 2 != 0;
          let numer = BigInt::from(4).pow(n as u32);
          let mut denom = BigInt::from(1);
          for i in (n + 1)..=(2 * n) {
            denom *= i;
          }
          return Ok(gamma_half_expr(numer, denom, is_neg));
        }
      }
      Ok(unevaluated("Gamma", args))
    }
    _ => {
      // Complex floating-point argument: use the Lanczos approximation.
      // Only triggers when the argument has a non-zero imaginary part AND
      // some component is a Real (not exact rationals/integers — those are
      // handled symbolically above).
      if let Some((re, im)) =
        crate::functions::math_ast::try_extract_complex_float(&args[0])
        && im != 0.0
        && contains_inexact_real(&args[0])
      {
        let (gr, gi) =
          crate::functions::math_ast::zeta_functions::gamma_complex(re, im);
        return Ok(crate::functions::math_ast::build_complex_float_expr(
          gr, gi,
        ));
      }
      Ok(unevaluated("Gamma", args))
    }
  }
}

fn gamma_half_expr(numer: BigInt, denom: BigInt, is_neg: bool) -> Expr {
  // Simplify the rational part
  let g = gcd_bigint(&numer, &denom);
  let num_simplified = &numer / &g;
  let den_simplified = &denom / &g;
  let coeff_num = if is_neg {
    -num_simplified.clone()
  } else {
    num_simplified.clone()
  };

  let sqrt_pi = Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![
      Expr::Identifier("Pi".to_string()),
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
      },
    ]
    .into(),
  };
  if den_simplified == BigInt::from(1) {
    if coeff_num == BigInt::from(1) {
      return sqrt_pi;
    }
    return Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![bigint_to_expr(coeff_num), sqrt_pi].into(),
    };
  }
  let coeff = Expr::FunctionCall {
    name: "Rational".to_string(),
    args: vec![bigint_to_expr(coeff_num), bigint_to_expr(den_simplified)]
      .into(),
  };
  Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![coeff, sqrt_pi].into(),
  }
}

/// Returns true if the expression contains a Real (machine-precision)
/// number anywhere in the tree. Used to distinguish exact symbolic
/// arguments (which Gamma should not evaluate numerically) from inexact
/// ones (which should be evaluated to floats).

/// Upper incomplete gamma function Gamma[a, z]
fn gamma_incomplete_upper(
  a: &Expr,
  z: &Expr,
) -> Result<Expr, InterpreterError> {
  // Gamma[a, 0] is the ordinary gamma function Gamma[a] when Re[a] > 0, and
  // diverges otherwise: Gamma[0, 0] = Infinity, Gamma[a, 0] = ComplexInfinity
  // for real a < 0. (Note this differs from Gamma[a] at non-positive a, where
  // the incomplete form diverges even though Gamma[a] itself may be finite.)
  if matches!(z, Expr::Integer(0))
    && let Some(a_val) = try_eval_to_f64(a)
  {
    if a_val > 0.0 {
      return gamma_ast(std::slice::from_ref(a));
    } else if a_val == 0.0 {
      return Ok(Expr::Identifier("Infinity".to_string()));
    } else {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
  }
  // Same divergences for an inexact zero z: Gamma[0, 0.] = Infinity and
  // Gamma[a, 0.] = ComplexInfinity for a < 0. (a > 0 falls through to the
  // numeric paths below, which return the finite Gamma[a] as a machine real,
  // so the numeric series never sees the z = 0 pole.)
  if matches!(z, Expr::Real(f) if *f == 0.0)
    && let Some(a_val) = try_eval_to_f64(a)
    && a_val <= 0.0
  {
    return Ok(if a_val == 0.0 {
      Expr::Identifier("Infinity".to_string())
    } else {
      Expr::Identifier("ComplexInfinity".to_string())
    });
  }

  // Special case: Gamma[1, z] = E^(-z)
  if matches!(a, Expr::Integer(1)) {
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![
        Expr::Identifier("E".to_string()),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), z.clone()].into(),
        },
      ]
      .into(),
    });
  }

  // For positive integer a with numeric z, prefer the exact closed form
  //   Gamma[n, z] = (n-1)! · E^{-z} · Σ_{k=0..n-1} z^k / k!
  // so e.g. `Gamma[3, 8] → 82/E^8` (matching Wolfram), rather than the
  // continued-fraction f64 approximation which loses exactness when both
  // args are integers.
  if let Some(n) = expr_to_i128(a)
    && n > 0
    && try_eval_to_f64(z).is_some()
  {
    return gamma_incomplete_upper_int_a(n, z);
  }

  // Numeric evaluation: both args are real numbers AND at least one is Real
  // (inexact). When both are integers/rationals, Wolfram keeps the form
  // symbolic — e.g. `Gamma[0, 1]` stays as `Gamma[0, 1]` rather than
  // collapsing to `0.21938…`.
  let has_real_arg = matches!(a, Expr::Real(_)) || matches!(z, Expr::Real(_));
  if has_real_arg
    && let (Some(a_val), Some(z_val)) = (try_eval_to_f64(a), try_eval_to_f64(z))
    && z_val >= 0.0
  {
    let result = upper_incomplete_gamma(a_val, z_val);
    if result.is_finite() {
      return Ok(Expr::Real(result));
    }
  }

  // Default: return unevaluated
  Ok(Expr::FunctionCall {
    name: "Gamma".to_string(),
    args: vec![a.clone(), z.clone()].into(),
  })
}

/// Build the closed form Gamma[n, z] = (n-1)! · E^{-z} · Σ z^k/k! for
/// positive integer n. Lets the evaluator simplify the resulting Times.
fn gamma_incomplete_upper_int_a(
  n: i128,
  z: &Expr,
) -> Result<Expr, InterpreterError> {
  let n = n as usize;
  let mut terms: Vec<Expr> = Vec::new();
  let mut factorial: i128 = 1;
  for k in 0..n {
    if k > 0 {
      factorial *= k as i128;
    }
    let z_power = if k == 0 {
      Expr::Integer(1)
    } else if k == 1 {
      z.clone()
    } else {
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![z.clone(), Expr::Integer(k as i128)].into(),
      }
    };
    let term = if factorial == 1 {
      z_power
    } else {
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(1), Expr::Integer(factorial)].into(),
          },
          z_power,
        ]
        .into(),
      }
    };
    terms.push(term);
  }
  let sum = if terms.len() == 1 {
    terms.into_iter().next().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    }
  };
  let mut n_minus_1_factorial: i128 = 1;
  for i in 2..n as i128 {
    n_minus_1_factorial *= i;
  }
  let exp_neg_z = Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![
      Expr::Identifier("E".to_string()),
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), z.clone()].into(),
      },
    ]
    .into(),
  };
  let result = if n_minus_1_factorial == 1 {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![exp_neg_z, sum].into(),
    }
  } else {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(n_minus_1_factorial), exp_neg_z, sum].into(),
    }
  };
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Numerical upper incomplete gamma via continued fraction (Legendre)
fn upper_incomplete_gamma(a: f64, z: f64) -> f64 {
  if z == 0.0 {
    return gamma_fn(a);
  }
  // Special case: Gamma[0, z] = E_1(z) (the exponential integral). The
  // generic series/CF route uses Gamma(a)·…−lower(a, z), which is
  // singular at a=0 (Gamma(0)=∞ minus finite); use a dedicated routine.
  if a == 0.0 && z > 0.0 {
    return exp_integral_e1(z);
  }
  // Use series for small z, continued fraction for large z
  if z < a + 1.0 {
    // Gamma(a, z) = Gamma(a) - gamma_lower(a, z)
    gamma_fn(a) - lower_incomplete_gamma_series(a, z)
  } else {
    // Continued fraction representation (Legendre)
    upper_incomplete_gamma_cf(a, z)
  }
}

/// E_1(z) = Gamma[0, z] for real z > 0.
/// Uses the convergent series for z ≤ 1 and a Lentz continued fraction
/// for larger z (cf. Numerical Recipes §6.3).
fn exp_integral_e1(z: f64) -> f64 {
  if z <= 1.0 {
    let euler_gamma = 0.5772156649015329_f64;
    let mut sum = -euler_gamma - z.ln();
    let mut term = 1.0_f64;
    for k in 1..200 {
      term *= -z / k as f64;
      let inc = -term / k as f64;
      sum += inc;
      if inc.abs() < 1e-18 * sum.abs().max(1.0) {
        break;
      }
    }
    sum
  } else {
    // Lentz continued fraction:
    //   E_1(z) = e^{-z} / (z + 1/(1 + 1/(z + 2/(1 + 2/(z + …)))))
    let mut b = z + 1.0;
    let mut c = 1e30_f64;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..200 {
      let i_f = i as f64;
      let a_term = -i_f * i_f;
      b += 2.0;
      d = 1.0 / (a_term * d + b);
      c = b + a_term / c;
      let del = c * d;
      h *= del;
      if (del - 1.0).abs() < 1e-15 {
        break;
      }
    }
    h * (-z).exp()
  }
}

/// Lower incomplete gamma via series expansion
fn lower_incomplete_gamma_series(a: f64, z: f64) -> f64 {
  let mut sum = 1.0 / a;
  let mut term = 1.0 / a;
  for n in 1..200 {
    term *= z / (a + n as f64);
    sum += term;
    if term.abs() < 1e-15 * sum.abs() {
      break;
    }
  }
  sum * (-z).exp() * z.powf(a)
}

/// Upper incomplete gamma via continued fraction
fn upper_incomplete_gamma_cf(a: f64, z: f64) -> f64 {
  // Modified Lentz's method. `c` starts at 1/FPMIN (a large value), not
  // FPMIN itself — initializing it small makes the first `an/c` term blow
  // up and the whole continued fraction diverges.
  let mut c = 1e30_f64;
  let mut d = z + 1.0 - a;
  if d.abs() < 1e-30 {
    d = 1e-30;
  }
  d = 1.0 / d;
  let mut f = d;
  for n in 1..200 {
    let an = n as f64 * (a - n as f64);
    let bn = z + (2 * n + 1) as f64 - a;
    d = bn + an * d;
    if d.abs() < 1e-30 {
      d = 1e-30;
    }
    c = bn + an / c;
    if c.abs() < 1e-30 {
      c = 1e-30;
    }
    d = 1.0 / d;
    let delta = c * d;
    f *= delta;
    if (delta - 1.0).abs() < 1e-15 {
      break;
    }
  }
  f * (-z).exp() * z.powf(a)
}

/// Lanczos approximation for the Gamma function
pub fn gamma_fn(x: f64) -> f64 {
  if x < 0.5 {
    // Reflection formula: Gamma(1-z) * Gamma(z) = pi / sin(pi*z)
    std::f64::consts::PI
      / ((std::f64::consts::PI * x).sin() * gamma_fn(1.0 - x))
  } else {
    let x = x - 1.0;
    let g = 7.0;
    let c = [
      0.999_999_999_999_809_9,
      676.5203681218851,
      -1259.1392167224028,
      771.323_428_777_653_1,
      -176.615_029_162_140_6,
      12.507343278686905,
      -0.13857109526572012,
      9.984_369_578_019_572e-6,
      1.5056327351493116e-7,
    ];
    let mut sum = c[0];
    for (i, &ci) in c.iter().enumerate().skip(1) {
      sum += ci / (x + i as f64);
    }
    let t = x + g + 0.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * sum
  }
}

/// Beta[a, b] - Euler beta function
/// Beta[a, b] = Gamma[a] * Gamma[b] / Gamma[a + b]
/// Beta[z, a, b] - incomplete Beta function
/// Beta[z, a, b] = integral_0^z t^(a-1) (1 - t)^(b-1) dt
/// Exact factorial as a BigInt, so Beta of larger integer arguments does not
/// overflow i128 (e.g. Beta[20, 20] needs 39! ≈ 2×10^46).
fn factorial_big(n: usize) -> BigInt {
  let mut result = BigInt::from(1);
  for i in 2..=n {
    result *= i as u64;
  }
  result
}

pub fn beta_parts_big(a: i128, b: i128) -> (BigInt, BigInt) {
  // Beta[a, b] = (a-1)! (b-1)! / (a+b-1)!
  // = (b-1)! / (a(a+1)...(a+b-1)) or
  // = (a-1)! / (b(b+1)...(a+b-1))
  // = (z-1)! / ((a+b-z)(a+b-z+1)...(a+b-1)) where z=min(a,b)
  let z = a.min(b);
  let mut num = BigInt::from(1);
  for i in 2..z {
    num *= i;
  }
  let mut den = BigInt::from(1);
  for i in (a + b - z)..(a + b) {
    den *= i;
  }
  (num, den)
}

pub fn beta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 3 {
    return incomplete_beta_ast(&args[0], &args[1], &args[2]);
  }
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Beta expects 2 or 3 arguments".into(),
    ));
  }

  // Try to evaluate for positive integer arguments
  // Beta[m, n] = (m-1)! * (n-1)! / (m+n-1)!
  if let (Expr::Integer(a), Expr::Integer(b)) = (&args[0], &args[1])
    && *a > 0
    && *b > 0
  {
    let (num, den) = beta_parts_big(*a, *b);
    return Ok(make_rational_expr(num, den));
  }

  // Integer arguments with at least one non-positive: resolve via the Gamma
  // pole structure of Beta[a, b] = Gamma[a] Gamma[b] / Gamma[a+b]. Each of
  // a <= 0 and b <= 0 contributes a numerator pole; a+b <= 0 contributes a
  // denominator pole. A surviving numerator pole gives ComplexInfinity; when
  // the poles exactly cancel the limit is finite.
  if let (Expr::Integer(a), Expr::Integer(b)) = (&args[0], &args[1]) {
    let (a, b) = (*a, *b);
    let num_poles = (a <= 0) as i32 + (b <= 0) as i32;
    if num_poles > 0 {
      let den_pole = (a + b <= 0) as i32;
      if num_poles - den_pole >= 1 {
        return Ok(Expr::Identifier("ComplexInfinity".to_string()));
      }
      // Poles cancel (exactly one argument non-positive and a+b <= 0): the
      // finite limit is (-1)^pos * (pos-1)! * (m - pos)! / m!, where `pos` is
      // the positive argument and m = -neg (the magnitude of the other one).
      // (pos-1)! (m-pos)! / m! = (pos-1)! ((m-(pos-1))-1)! / ((pos-1) + (m-(pos-1)))!
      // = Beta[pos, m-pos+1]
      let (pos, neg) = if a > 0 { (a, b) } else { (b, a) };
      let m = -neg;
      let is_neg = pos % 2 != 0;
      let (mut num, den) = beta_parts_big(pos, m - pos + 1);
      num = if is_neg { -num } else { num };
      return Ok(make_rational_expr(num, den));
    }
  }

  // Beta[a, n] (or Beta[n, a]) with n a positive integer and a an exact
  // non-integer rational: Beta[a, b] = Gamma[a] Gamma[b] / Gamma[a+b], which
  // for an integer second argument telescopes to the rising factorial
  //   Beta[a, n] = (n-1)! / Pochhammer[a, n].
  // This yields an exact rational for any rational a (e.g. Beta[1/2, 3] = 16/15,
  // Beta[7/3, 2] = 9/70) and complements the integer/integer and
  // half-integer/half-integer branches handled above.
  {
    let is_pos_int = |e: &Expr| matches!(e, Expr::Integer(n) if *n > 0);
    let is_exact_rational = |e: &Expr| matches!(e, Expr::FunctionCall { name, .. } if name == "Rational");
    for (x, n_arg) in [(&args[0], &args[1]), (&args[1], &args[0])] {
      if let Expr::Integer(n) = n_arg
        && is_pos_int(n_arg)
        && is_exact_rational(x)
      {
        let poch = pochhammer_ast(&[x.clone(), Expr::Integer(*n)])?;
        let fact = crate::evaluator::evaluate_function_call_ast(
          "Factorial",
          &[Expr::Integer(*n - 1)],
        )?;
        return crate::evaluator::evaluate_function_call_ast(
          "Divide",
          &[fact, poch],
        );
      }
    }
  }

  // Try rational args for half-integer cases
  // Beta[p/q, r/s] for half-integers involves Gamma at half-integers
  if let (Some(a_f), Some(b_f)) = (expr_to_f64(&args[0]), expr_to_f64(&args[1]))
  {
    // Check if both are positive half-integers (n + 1/2)
    let a_half = a_f * 2.0;
    let b_half = b_f * 2.0;
    if a_half == a_half.round()
      && b_half == b_half.round()
      && a_f > 0.0
      && b_f > 0.0
      && a_half.fract() == 0.0
      && b_half.fract() == 0.0
    {
      let a2 = a_half as i128;
      let b2 = b_half as i128;

      // At least one must be odd (half-integer) for Pi to appear
      if a2 % 2 != 0 || b2 % 2 != 0 {
        // Check if both args are Real (then numeric)
        if matches!(&args[0], Expr::Real(_))
          || matches!(&args[1], Expr::Real(_))
        {
          let result = gamma_fn(a_f) * gamma_fn(b_f) / gamma_fn(a_f + b_f);
          return Ok(Expr::Real(result));
        }
        // For exact half-integer arguments, compute via Gamma
        // If sum is integer, result is rational * sqrt(pi) or rational * pi
        let sum2 = a2 + b2;
        if sum2 % 2 == 0 {
          // Both half-integers or both integers, sum is integer
          // Result involves sqrt(pi) terms that may cancel
          // Use numeric for now unless both are half-integers with integer sum
          // Beta[a, b] = Γ(a)Γ(b)/Γ(a+b)
          // When a, b are half-integers, Γ(n+1/2) = (2n)! sqrt(π) / (4^n n!)
          // So Gamma product has π, and if sum is integer, Γ(sum) is (sum-1)!
          // Beta = Γ(a)Γ(b) / (sum-1)!
          let sum_int = (sum2 / 2) as usize;
          {
            let sum_fact = factorial_big(sum_int - 1);
            // Compute Γ(a) * Γ(b) / (sum-1)! where a, b are half-integers
            // Γ(k/2) for odd k: Γ((2m+1)/2) = (2m)! π^{1/2} / (4^m m!)
            // For even k: Γ(k/2) = ((k/2)-1)!. BigInt throughout so large
            // half-integers (e.g. Beta[21/2, 21/2]) don't overflow i128.
            let gamma_a = gamma_half_integer_parts_big(a2);
            let gamma_b = gamma_half_integer_parts_big(b2);
            if let (
              Some((a_num, a_den, a_pi_pow)),
              Some((b_num, b_den, b_pi_pow)),
            ) = (gamma_a, gamma_b)
            {
              let total_pi_pow = a_pi_pow + b_pi_pow; // each half-integer contributes 1/2
              let num = a_num * b_num;
              let den = a_den * b_den * sum_fact;
              let coeff = make_rational_expr(num, den);

              if total_pi_pow == 0 {
                return Ok(coeff);
              } else if total_pi_pow == 2 {
                // Two sqrt(Pi) factors = Pi → result is (num/den) * Pi.
                if matches!(&coeff, Expr::Integer(1)) {
                  return Ok(Expr::Identifier("Pi".to_string()));
                }
                return crate::evaluator::evaluate_expr_to_expr(
                  &Expr::BinaryOp {
                    op: BinaryOperator::Times,
                    left: Box::new(coeff),
                    right: Box::new(Expr::Identifier("Pi".to_string())),
                  },
                );
              }
            }
          }
        }
      }
    }
  }

  // Numeric evaluation
  if let (Some(a_f), Some(b_f)) = (expr_to_f64(&args[0]), expr_to_f64(&args[1]))
    && (matches!(&args[0], Expr::Real(_)) || matches!(&args[1], Expr::Real(_)))
  {
    // Beta[a, b] = Gamma[a] Gamma[b] / Gamma[a+b]. A non-positive integer a or
    // b is a numerator pole; a non-positive integer a+b is a denominator pole.
    // gamma_fn diverges/garbages at the poles, so resolve them by pole order
    // (mirroring the exact-integer branch above) before the numeric formula.
    let is_nonpos_int = |x: f64| {
      let r = x.round();
      r <= 0.0 && (x - r).abs() < 1e-9
    };
    let num_poles = is_nonpos_int(a_f) as i32 + is_nonpos_int(b_f) as i32;
    let den_pole = is_nonpos_int(a_f + b_f) as i32;
    let net = num_poles - den_pole;
    if net > 0 {
      // A surviving numerator pole → ComplexInfinity.
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    if net < 0 {
      // The denominator pole dominates → 0.
      return Ok(Expr::Real(0.0));
    }
    if num_poles > 0 {
      // net == 0 with poles present: they cancel to a finite value. Both
      // arguments are integers here, so the exact integer branch yields the
      // limit; convert it to a machine real to match the inexact inputs.
      let exact = beta_ast(&[
        Expr::Integer(a_f.round() as i128),
        Expr::Integer(b_f.round() as i128),
      ])?;
      if let Some(v) = expr_to_f64(&exact) {
        return Ok(Expr::Real(v));
      }
    }
    let result = gamma_fn(a_f) * gamma_fn(b_f) / gamma_fn(a_f + b_f);
    return Ok(Expr::Real(result));
  }

  // Unevaluated
  Ok(unevaluated("Beta", args))
}

/// Beta[z, a, b] — incomplete Beta function.
///
/// For positive integer b, expand (1 - t)^(b-1) by the binomial theorem
/// and integrate term-by-term:
///   B(z; a, b) = Σ_{k=0..b-1} C(b-1, k) · (-1)^k · z^(a+k) / (a+k)
/// This yields an exact rational (or closed-form symbolic) result for
/// rational a and z.
///
/// For other numeric inputs, fall back to the closed-form
/// Beta[z, a, b] = z^a · Hypergeometric2F1[a, 1 - b, a + 1, z] / a.
fn incomplete_beta_ast(
  z: &Expr,
  a: &Expr,
  b: &Expr,
) -> Result<Expr, InterpreterError> {
  // Special case: Beta[1, a, b] = Beta[a, b] (complete). wolframscript only
  // performs this reduction when both parameters are numeric; for a symbolic a
  // or b it keeps the three-argument form (e.g. Beta[1, 2, b] stays held).
  if matches!(z, Expr::Integer(1)) || matches!(z, Expr::Real(f) if *f == 1.0) {
    let is_num = |e: &Expr| {
      matches!(e, Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_))
        || matches!(e, Expr::FunctionCall { name, .. } if name == "Rational")
    };
    if is_num(a) && is_num(b) {
      let base = beta_ast(&[a.clone(), b.clone()])?;
      if matches!(z, Expr::Real(_)) {
        return crate::evaluator::evaluate_function_call_ast("N", &[base]);
      }
      return Ok(base);
    }
    return Ok(Expr::FunctionCall {
      name: "Beta".to_string(),
      args: vec![z.clone(), a.clone(), b.clone()].into(),
    });
  }

  // The polynomial closed-form is only available when z is numeric and b is a
  // positive whole number (exact integer or a whole-valued machine real, as
  // arises from N[…]). Symbolic z is left unevaluated, matching wolframscript.
  let z_is_numeric =
    matches!(z, Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_))
      || matches!(z, Expr::FunctionCall { name, .. } if name == "Rational");

  // Symbolic z has an elementary closed form only when a or b equals 1
  // (integrating t^(a-1) (1-t)^(b-1) collapses):
  //   Beta[z, 1, b] = (1 - (1 - z)^b)/b
  //   Beta[z, a, 1] = (z^a - 0^a)/a   (the lower-limit boundary term 0^a is
  //                                    kept symbolic, matching wolframscript).
  if !z_is_numeric {
    let minus = |x: Expr, y: Expr| Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(x),
      right: Box::new(y),
    };
    let power = |x: Expr, y: Expr| Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(x),
      right: Box::new(y),
    };
    let divide = |x: Expr, y: Expr| Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(x),
      right: Box::new(y),
    };
    if matches!(a, Expr::Integer(1)) {
      let num = minus(
        Expr::Integer(1),
        power(minus(Expr::Integer(1), z.clone()), b.clone()),
      );
      return crate::evaluator::evaluate_expr_to_expr(&divide(num, b.clone()));
    }
    if matches!(b, Expr::Integer(1)) {
      let num = minus(
        power(z.clone(), a.clone()),
        power(Expr::Integer(0), a.clone()),
      );
      return crate::evaluator::evaluate_expr_to_expr(&divide(num, a.clone()));
    }
  }
  let b_whole: Option<i128> = match b {
    Expr::Integer(n) if *n > 0 => Some(*n),
    Expr::Real(f) if *f > 0.0 && f.fract() == 0.0 => Some(*f as i128),
    _ => None,
  };

  if let Some(b_i) = b_whole
    && z_is_numeric
  {
    // B(z; a, b) = Σ_{k=0..b-1} C(b-1, k) (-1)^k z^(a+k) / (a+k)
    // (binomial expansion of (1 - t)^(b-1), integrated term by term).
    let mut term: Vec<Expr> = Vec::new();
    for k in 0..b_i {
      let coeff = crate::functions::binomial_coeff(b_i - 1, k);
      let sign = if k % 2 == 0 { 1 } else { -1 };
      // (a + k)
      let denom = crate::evaluator::evaluate_function_call_ast(
        "Plus",
        &[a.clone(), Expr::Integer(k)],
      )?;
      // z^(a + k)
      let z_pow = crate::evaluator::evaluate_function_call_ast(
        "Power",
        &[z.clone(), denom.clone()],
      )?;
      let signed_coeff = make_rational(sign * coeff, 1);
      // coeff * z^(a+k) / (a+k)
      let numer = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[signed_coeff, z_pow],
      )?;
      // This closed form is an internal numerical routine, not a user-level
      // Divide: when both operands are machine reals compute the quotient with a
      // plain (single-rounded) division. The language-level Divide models
      // wolframscript's Times[a, Power[b, -1]] multiply-by-reciprocal, which
      // would add a second rounding here and drift the incomplete-Beta value one
      // ULP away from wolframscript (e.g. N[Beta[2, 1/2, 3]]).
      let summand = match (&numer, &denom) {
        (Expr::Real(nf), Expr::Real(df)) => Expr::Real(nf / df),
        _ => crate::evaluator::evaluate_function_call_ast(
          "Divide",
          &[numer, denom],
        )?,
      };
      term.push(summand);
    }
    let closed = crate::evaluator::evaluate_function_call_ast("Plus", &term)?;

    // Any inexact argument ⇒ machine-number result.
    let any_inexact = contains_inexact_real(z)
      || contains_inexact_real(a)
      || contains_inexact_real(b);
    if any_inexact {
      return crate::evaluator::evaluate_function_call_ast("N", &[closed]);
    }

    // Fully exact input: wolframscript only auto-evaluates the incomplete Beta
    // to a closed form when both a and b are (positive) integers. For a
    // non-integer exact a (e.g. Beta[2, 1/2, 3]) it stays symbolic, even though
    // the rational closed form exists. Integer a (including non-positive a,
    // which yields ComplexInfinity through the pole at a + k = 0) matches WL.
    if matches!(a, Expr::Integer(_)) {
      return Ok(closed);
    }
  }

  // Otherwise leave unevaluated — matches wolframscript for symbolic z, a
  // non-whole b, or an exact non-integer a.
  Ok(Expr::FunctionCall {
    name: "Beta".to_string(),
    args: vec![z.clone(), a.clone(), b.clone()].into(),
  })
}

/// Stirling's series for log(gamma(z)) when z is large and positive.
/// log(gamma(z)) = (z - 1/2)*log(z) - z + (1/2)*log(2π)
///                 + 1/(12z) - 1/(360z³) + 1/(1260z⁵) - ...
fn log_gamma_stirling(z: f64) -> f64 {
  use std::f64::consts::PI;
  let log_2pi = (2.0 * PI).ln();
  let z2 = z * z;
  let z3 = z2 * z;
  let z5 = z3 * z2;
  let z7 = z5 * z2;
  (z - 0.5) * z.ln() - z + 0.5 * log_2pi + 1.0 / (12.0 * z) - 1.0 / (360.0 * z3)
    + 1.0 / (1260.0 * z5)
    - 1.0 / (1680.0 * z7)
}

/// LogGamma[z] — logarithm of the gamma function.
pub fn log_gamma_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(unevaluated("LogGamma", args));
  }

  let z = &args[0];

  // Handle exact integer cases
  if let Some(n) = expr_to_i128(z) {
    if n <= 0 {
      // LogGamma[0] = LogGamma[-n] = Infinity
      return Ok(Expr::Identifier("Infinity".to_string()));
    }
    if n == 1 || n == 2 {
      return Ok(Expr::Integer(0)); // Log[0!] = Log[1!] = 0
    }
    // LogGamma[n] = Log[(n-1)!]
    let gamma_result = gamma_ast(&[z.clone()])?;
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Log".to_string(),
      args: vec![gamma_result].into(),
    });
  }

  // Handle Rational arguments — compute Gamma then Log
  if let Expr::FunctionCall { name, args: fargs } = z
    && name == "Rational"
    && fargs.len() == 2
    && let (Expr::Integer(n), Expr::Integer(d)) = (&fargs[0], &fargs[1])
  {
    if *d == 2 && *n > 0 {
      // Half-integer: LogGamma[k/2] = Log[Gamma[k/2]]
      let gamma_result = gamma_ast(&[z.clone()])?;
      return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![gamma_result].into(),
      });
    }
    if *n <= 0 && *d > 0 && *n % *d == 0 {
      // Non-positive integer
      return Ok(Expr::Identifier("Infinity".to_string()));
    }
  }

  // Handle numeric (Real) — use lgamma
  if let Some(f) = try_eval_to_f64(z)
    && (matches!(z, Expr::Real(_))
      || matches!(z, Expr::FunctionCall { name, .. } if name == "Rational")
        && try_eval_to_f64(z).is_some())
  {
    if f <= 0.0 && f == f.floor() {
      return Ok(Expr::Identifier("Infinity".to_string()));
    }
    if matches!(z, Expr::Real(_)) {
      // Compute log(|gamma(f)|) directly to avoid overflow for large f.
      // Use Stirling series for f > 12, and Log[Gamma[f]] for smaller.
      let result = if f >= 12.0 {
        log_gamma_stirling(f)
      } else {
        gamma_fn(f).abs().ln()
      };
      return Ok(Expr::Real(result));
    }
  }

  // Complex (inexact) numeric path: route to complex LogGamma using
  // forward-shift + Stirling series. Wolfram's LogGamma is the analytic
  // continuation, which can differ from Log[Gamma[z]] by a 2πi multiple
  // for z to the left of the imaginary axis.
  if let Some((re, im)) =
    crate::functions::math_ast::try_extract_complex_float(z)
    && im != 0.0
    && contains_inexact_real(z)
  {
    let (lr, li) =
      crate::functions::math_ast::zeta_functions::log_gamma_complex(re, im);
    return Ok(crate::functions::math_ast::build_complex_float_expr(lr, li));
  }

  // Return unevaluated for symbolic case
  Ok(unevaluated("LogGamma", args))
}

/// Compute parts of Gamma at half-integer: Gamma(k/2) for integer k > 0
/// Returns (num, den, pi_power) where result = (num/den) * Pi^(pi_power/2)
/// pi_power is 0 or 1 (representing sqrt(Pi)^pi_power).
/// Use BigInt so half-integer Beta arguments do not overflow i128
/// (e.g. Beta[21/2, 21/2]).
fn gamma_half_integer_parts_big(k2: i128) -> Option<(BigInt, BigInt, i128)> {
  if k2 <= 0 {
    return None;
  }
  if k2 % 2 == 0 {
    let m = (k2 / 2) as usize;
    Some((factorial_big(m - 1), BigInt::from(1), 0))
  } else {
    let m = ((k2 - 1) / 2) as usize;
    let two_m_fact = factorial_big(2 * m);
    let m_fact = factorial_big(m);
    let four_m = BigInt::from(4).pow(m as u32);
    Some((two_m_fact, four_m * m_fact, 1))
  }
}

/// BetaRegularized[z, a, b] - Regularized incomplete beta function I_z(a, b)
/// I_z(a, b) = B(z; a, b) / B(a, b) where B(z; a, b) = ∫₀ᶻ t^(a-1) (1-t)^(b-1) dt
pub fn beta_regularized_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "BetaRegularized expects exactly 3 arguments".into(),
    ));
  }

  let z_expr = &args[0];
  let a_expr = &args[1];
  let b_expr = &args[2];

  // BetaRegularized[0, a, b] = 0 (when a > 0)
  if is_expr_zero(z_expr) && is_positive_numeric(a_expr) {
    return Ok(Expr::Integer(0));
  }

  // BetaRegularized[1, a, b] = 1 (when b > 0)
  if matches!(z_expr, Expr::Integer(1)) && is_positive_numeric(b_expr) {
    return Ok(Expr::Integer(1));
  }

  // Elementary closed forms when one shape parameter is 1 (matching
  // wolframscript), valid for any b / a, including symbolic ones:
  //   I_z(1, b) = 1 - (1 - z)^b
  //   I_z(a, 1) = z^a - 0^a   (= z^a for a > 0; stays -0^a + z^a for symbolic a)
  // A machine-Real z is left to the numeric path below.
  if !matches!(z_expr, Expr::Real(_)) {
    let times = |a: Expr, b: Expr| Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![a, b].into(),
    };
    let power = |a: Expr, b: Expr| Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![a, b].into(),
    };
    let plus = |a: Expr, b: Expr| Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![a, b].into(),
    };
    // a == 1 takes precedence (so a == b == 1 reduces to z).
    if matches!(a_expr, Expr::Integer(1)) {
      // 1 - (1 - z)^b
      let one_minus_z =
        plus(Expr::Integer(1), times(Expr::Integer(-1), z_expr.clone()));
      let result = plus(
        Expr::Integer(1),
        times(Expr::Integer(-1), power(one_minus_z, b_expr.clone())),
      );
      return crate::evaluator::evaluate_expr_to_expr(&result);
    }
    // Float a == 1. expands the same way but with a machine-real
    // leading 1. and an exact inner (1 - z), matching wolframscript:
    // BetaRegularized[z, 1., 2.] → 1. - (1 - z)^2.
    if matches!(a_expr, Expr::Real(r) if *r == 1.0) {
      let one_minus_z =
        plus(Expr::Integer(1), times(Expr::Integer(-1), z_expr.clone()));
      let result = plus(
        Expr::Real(1.0),
        times(Expr::Integer(-1), power(one_minus_z, b_expr.clone())),
      );
      return crate::evaluator::evaluate_expr_to_expr(&result);
    }
    if matches!(b_expr, Expr::Real(r) if *r == 1.0) {
      // wolframscript shows 0. + z^a here; Woxi's Plus folds the
      // machine-real zero away, so this renders as plain z^a
      // (documented divergence).
      let result = plus(Expr::Real(0.0), power(z_expr.clone(), a_expr.clone()));
      return crate::evaluator::evaluate_expr_to_expr(&result);
    }
    if matches!(b_expr, Expr::Integer(1)) {
      // z^a - 0^a
      let result = plus(
        power(z_expr.clone(), a_expr.clone()),
        times(Expr::Integer(-1), power(Expr::Integer(0), a_expr.clone())),
      );
      return crate::evaluator::evaluate_expr_to_expr(&result);
    }
  }

  // Numeric evaluation when all arguments are numeric and at least one is Real
  let z_val = expr_to_f64(z_expr);
  let a_val = expr_to_f64(a_expr);
  let b_val = expr_to_f64(b_expr);
  let has_real = matches!(z_expr, Expr::Real(_))
    || matches!(a_expr, Expr::Real(_))
    || matches!(b_expr, Expr::Real(_));

  if let (Some(z), Some(a), Some(b)) = (z_val, a_val, b_val)
    && has_real
  {
    return Ok(Expr::Real(beta_regularized_numeric(z, a, b)));
  }

  // Exact evaluation for positive integer a, b and exact rational z:
  // I_z(a, b) = sum_{j=a}^{a+b-1} C(a+b-1, j) z^j (1-z)^(a+b-1-j)
  // (the binomial-tail identity; polynomial in z, so exact)
  if let (Expr::Integer(a), Expr::Integer(b)) = (a_expr, b_expr)
    && *a >= 1
    && *b >= 1
    && *a + *b <= 64
    && (matches!(z_expr, Expr::Integer(_))
      || matches!(z_expr, Expr::FunctionCall { name, .. } if name == "Rational"))
  {
    let n = *a + *b - 1;
    let one_minus_z = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        Expr::Integer(1),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), z_expr.clone()].into(),
        },
      ]
      .into(),
    };
    let terms: Vec<Expr> = (*a..=n)
      .map(|j| Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::Integer(crate::functions::binomial_coeff(n, j)),
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![z_expr.clone(), Expr::Integer(j)].into(),
          },
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![one_minus_z.clone(), Expr::Integer(n - j)].into(),
          },
        ]
        .into(),
      })
      .collect();
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    });
  }

  // Unevaluated
  Ok(unevaluated("BetaRegularized", args))
}

/// Check if an expression is a positive number
fn is_positive_numeric(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(n) => *n > 0,
    Expr::Real(x) => *x > 0.0,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        (*n > 0 && *d > 0) || (*n < 0 && *d < 0)
      } else {
        false
      }
    }
    _ => false,
  }
}

/// Compute the regularized incomplete beta function I_x(a, b) numerically
/// Uses the continued fraction representation (Lentz's algorithm)
fn beta_regularized_numeric(x: f64, a: f64, b: f64) -> f64 {
  if x <= 0.0 {
    return 0.0;
  }
  if x >= 1.0 {
    return 1.0;
  }

  // Use symmetry relation: I_x(a,b) = 1 - I_{1-x}(b,a) when x > (a+1)/(a+b+2)
  // This ensures convergence of the continued fraction
  if x > (a + 1.0) / (a + b + 2.0) {
    return 1.0 - beta_regularized_numeric(1.0 - x, b, a);
  }

  // Compute using the continued fraction representation
  // I_x(a,b) = x^a * (1-x)^b / (a * B(a,b)) * 1/(1+ d1/(1+ d2/(1+ ...)))
  // where d_{2m+1} = -(a+m)(a+b+m) x / ((a+2m)(a+2m+1))
  //       d_{2m}   = m(b-m) x / ((a+2m-1)(a+2m))

  let ln_prefix = a * x.ln() + b * (1.0 - x).ln() - a.ln() - ln_beta(a, b);
  let prefix = ln_prefix.exp();

  // Lentz's continued fraction method
  let mut c = 1.0_f64;
  let mut d = 1.0 - (a + b) * x / (a + 1.0);
  if d.abs() < 1e-30 {
    d = 1e-30;
  }
  d = 1.0 / d;
  let mut result = d;

  for m in 1..200 {
    let m_f = m as f64;

    // Even step: d_{2m}
    let numerator =
      m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
    d = 1.0 + numerator * d;
    if d.abs() < 1e-30 {
      d = 1e-30;
    }
    c = 1.0 + numerator / c;
    if c.abs() < 1e-30 {
      c = 1e-30;
    }
    d = 1.0 / d;
    result *= d * c;

    // Odd step: d_{2m+1}
    let numerator = -(a + m_f) * (a + b + m_f) * x
      / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
    d = 1.0 + numerator * d;
    if d.abs() < 1e-30 {
      d = 1e-30;
    }
    c = 1.0 + numerator / c;
    if c.abs() < 1e-30 {
      c = 1e-30;
    }
    d = 1.0 / d;
    let delta = d * c;
    result *= delta;

    if (delta - 1.0).abs() < 1e-15 {
      break;
    }
  }

  prefix * result
}

/// Numeric inverse of the regularized incomplete beta function: returns z in
/// [0, 1] such that `beta_regularized_numeric(z, a, b) == s`. Because I_z(a, b)
/// increases monotonically from 0 (at z = 0) to 1 (at z = 1), the root is found
/// by bisection.
fn inverse_beta_regularized_numeric(s: f64, a: f64, b: f64) -> f64 {
  if s <= 0.0 {
    return 0.0;
  }
  if s >= 1.0 {
    return 1.0;
  }
  let mut lo = 0.0_f64;
  let mut hi = 1.0_f64;
  for _ in 0..200 {
    let mid = 0.5 * (lo + hi);
    if beta_regularized_numeric(mid, a, b) < s {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  0.5 * (lo + hi)
}

/// InverseBetaRegularized[s, a, b] - inverse of the regularized incomplete beta
/// function: returns z with BetaRegularized[z, a, b] == s. Exact non-elementary
/// inputs stay symbolic (wolframscript returns an algebraic Root object there);
/// inexact (machine-real) inputs evaluate numerically by bisection.
pub fn inverse_beta_regularized_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let symbolic = || Ok(unevaluated("InverseBetaRegularized", args));

  // Generalized 4-arg form InverseBetaRegularized[z0, s, a, b] solves
  // BetaRegularized[z0, z, a, b] = I_z(a, b) - I_z0(a, b) == s for z, i.e.
  // I_z(a, b) = s + I_z0(a, b). Exact arguments stay symbolic (matching
  // wolframscript); inexact arguments evaluate numerically.
  if args.len() == 4 {
    let inexact = args.iter().any(|e| matches!(e, Expr::Real(_)));
    if let (true, Some(z0), Some(sv), Some(av), Some(bv)) = (
      inexact,
      expr_to_f64(&args[0]),
      expr_to_f64(&args[1]),
      expr_to_f64(&args[2]),
      expr_to_f64(&args[3]),
    ) && av > 0.0
      && bv > 0.0
    {
      let target = sv + beta_regularized_numeric(z0, av, bv);
      return Ok(Expr::Real(inverse_beta_regularized_numeric(target, av, bv)));
    }
    return symbolic();
  }
  if args.len() != 3 {
    return symbolic();
  }

  let s = &args[0];
  let a = &args[1];
  let b = &args[2];
  let s_num = expr_to_f64(s);
  let a_num = expr_to_f64(a);
  let b_num = expr_to_f64(b);

  // Boundary values for numeric a, b: s == 0 -> 0, s == 1 -> 1.
  if a_num.is_some() && b_num.is_some() {
    if matches!(s, Expr::Integer(0)) {
      return Ok(Expr::Integer(0));
    }
    if matches!(s, Expr::Integer(1)) {
      return Ok(Expr::Integer(1));
    }
  }

  // I_z(1, 1) = z, so the inverse is the identity for a numeric s.
  if matches!(a, Expr::Integer(1))
    && matches!(b, Expr::Integer(1))
    && s_num.is_some()
  {
    return Ok(s.clone());
  }

  // Inexact (machine-real) input -> numeric bisection.
  let inexact = matches!(s, Expr::Real(_))
    || matches!(a, Expr::Real(_))
    || matches!(b, Expr::Real(_));
  if let (true, Some(sv), Some(av), Some(bv)) = (inexact, s_num, a_num, b_num)
    && av > 0.0
    && bv > 0.0
  {
    return Ok(Expr::Real(inverse_beta_regularized_numeric(sv, av, bv)));
  }

  symbolic()
}

/// Compute ln(Beta(a, b)) = ln(Gamma(a)) + ln(Gamma(b)) - ln(Gamma(a+b))
fn ln_beta(a: f64, b: f64) -> f64 {
  lgamma(a) + lgamma(b) - lgamma(a + b)
}

/// Log-gamma function using Lanczos approximation
pub fn lgamma(x: f64) -> f64 {
  // Use the standard library's ln_gamma if available via f64 methods
  // Otherwise use Lanczos approximation
  let g = 7.0;
  let coefs = [
    0.999_999_999_999_809_9,
    676.5203681218851,
    -1259.1392167224028,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507343278686905,
    -0.13857109526572012,
    9.984_369_578_019_572e-6,
    1.5056327351493116e-7,
  ];

  if x < 0.5 {
    // Reflection formula: Gamma(1-x) * Gamma(x) = pi / sin(pi*x)
    let log_pi = std::f64::consts::PI.ln();
    log_pi - (std::f64::consts::PI * x).sin().abs().ln() - lgamma(1.0 - x)
  } else {
    let x = x - 1.0;
    let mut base = coefs[0];
    for (i, &c) in coefs.iter().enumerate().skip(1) {
      base += c / (x + i as f64);
    }
    let t = x + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (t.ln() * (x + 0.5)) - t
      + base.ln()
  }
}

/// GammaRegularized[a, z] - Regularized upper incomplete gamma function Q(a, z)
/// Q(a, z) = Gamma(a, z) / Gamma(a) = 1 - P(a, z)
pub fn gamma_regularized_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "GammaRegularized expects exactly 2 arguments".into(),
    ));
  }

  let a_expr = &args[0];
  let z_expr = &args[1];

  // GammaRegularized[0, z] = Gamma[0, z]/Gamma[0] = 0 for any z (Gamma[0] is
  // ComplexInfinity), matching wolframscript — including z == 0.
  if matches!(a_expr, Expr::Integer(0)) {
    return Ok(Expr::Integer(0));
  }

  // GammaRegularized[a, 0] = 1
  if is_expr_zero(z_expr) && is_positive_numeric(a_expr) {
    return Ok(Expr::Integer(1));
  }

  // GammaRegularized[a, Infinity] = 0
  if matches!(z_expr, Expr::Identifier(s) if s == "Infinity") {
    return Ok(Expr::Integer(0));
  }

  // Numeric evaluation
  let a_val = expr_to_f64(a_expr);
  let z_val = expr_to_f64(z_expr);
  let has_real =
    matches!(a_expr, Expr::Real(_)) || matches!(z_expr, Expr::Real(_));

  if let (Some(a), Some(z)) = (a_val, z_val)
    && has_real
  {
    return Ok(Expr::Real(gamma_regularized_numeric(a, z)));
  }

  // GammaRegularized[1, z] = E^(-z) for any z (Wolfram auto-evaluates
  // this case even symbolically)
  if matches!(a_expr, Expr::Integer(1)) {
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![
        Expr::Constant("E".to_string()),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), z_expr.clone()].into(),
        },
      ]
      .into(),
    });
  }

  // Exact evaluation for positive integer a with exact numeric z:
  // Q(m, z) = e^(-z) sum_{k<m} z^k/k!
  if let Expr::Integer(m) = a_expr
    && *m >= 1
    && *m <= 64
    && (matches!(z_expr, Expr::Integer(_))
      || matches!(z_expr, Expr::FunctionCall { name, .. } if name == "Rational"))
  {
    let mut terms: Vec<Expr> = Vec::new();
    let mut k_fact = 1i128;
    for k in 0..*m {
      if k > 0 {
        k_fact *= k;
      }
      let z_pow = Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![z_expr.clone(), Expr::Integer(k)].into(),
      };
      terms.push(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(1), Expr::Integer(k_fact)].into(),
          },
          z_pow,
        ]
        .into(),
      });
    }
    let e_pow = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![
        Expr::Constant("E".to_string()),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), z_expr.clone()].into(),
        },
      ]
      .into(),
    };
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        e_pow,
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: terms.into(),
        },
      ]
      .into(),
    });
  }

  // Unevaluated
  Ok(unevaluated("GammaRegularized", args))
}

/// MarcumQ[m, a, b] / MarcumQ[m, a, b0, b1] — generalized Marcum Q.
/// Exact arguments stay symbolic apart from the wolframscript rules:
/// b = 0 gives 1 for positive numeric m (ComplexInfinity at
/// non-positive integers, 1 - E^(-a^2/2) at m = 0) and a = 0 reduces
/// to GammaRegularized[m, b^2/2]. Machine reals evaluate via the
/// Poisson-weighted incomplete-gamma series; the four-argument form is
/// the difference Q(m,a,b0) - Q(m,a,b1).
pub fn marcum_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated("MarcumQ", args);
  if args.len() != 3 && args.len() != 4 {
    return Ok(unevaluated(args));
  }
  let numeric_value = |e: &Expr| -> Option<f64> {
    match e {
      Expr::Integer(v) => Some(*v as f64),
      Expr::Real(v) => Some(*v),
      Expr::FunctionCall { name, args } if name == "Rational" => {
        match (&args[0], &args[1]) {
          (Expr::Integer(p), Expr::Integer(q)) if *q != 0 => {
            Some(*p as f64 / *q as f64)
          }
          _ => None,
        }
      }
      _ => crate::functions::math_ast::numeric_utils::try_eval_to_f64(e),
    }
  };
  let has_real = args.iter().any(|e| matches!(e, Expr::Real(_)));

  if args.len() == 4 {
    if has_real {
      let vals: Vec<Option<f64>> = args.iter().map(numeric_value).collect();
      if let [Some(m), Some(a), Some(b0), Some(b1)] = vals[..] {
        return Ok(Expr::Real(
          marcum_q_numeric(m, a, b0) - marcum_q_numeric(m, a, b1),
        ));
      }
    }
    return Ok(unevaluated(args));
  }

  let (m, a, b) = (&args[0], &args[1], &args[2]);
  // b = 0
  if is_expr_zero(b) {
    if is_positive_numeric(m) {
      return Ok(Expr::Integer(1));
    }
    if matches!(m, Expr::Integer(v) if *v < 0) {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    if matches!(m, Expr::Integer(0)) {
      // 1 - E^(-a^2/2)
      return Ok(Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          Expr::Integer(1),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::Integer(-1),
              Expr::FunctionCall {
                name: "Power".to_string(),
                args: vec![
                  Expr::Constant("E".to_string()),
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: vec![
                      Expr::FunctionCall {
                        name: "Rational".to_string(),
                        args: vec![Expr::Integer(-1), Expr::Integer(2)].into(),
                      },
                      Expr::FunctionCall {
                        name: "Power".to_string(),
                        args: vec![a.clone(), Expr::Integer(2)].into(),
                      },
                    ]
                    .into(),
                  },
                ]
                .into(),
              },
            ]
            .into(),
          },
        ]
        .into(),
      });
    }
    return Ok(unevaluated(args));
  }
  // a = 0: GammaRegularized[m, b^2/2]
  if is_expr_zero(a) {
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "GammaRegularized".to_string(),
      args: vec![
        m.clone(),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
            },
            Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![b.clone(), Expr::Integer(2)].into(),
            },
          ]
          .into(),
        },
      ]
      .into(),
    });
  }
  if has_real
    && let (Some(m), Some(a), Some(b)) =
      (numeric_value(m), numeric_value(a), numeric_value(b))
  {
    return Ok(Expr::Real(marcum_q_numeric(m, a, b)));
  }
  Ok(unevaluated(args))
}

/// Q_m(a, b) = sum_n Poisson(n; a^2/2) Q(m + n, b^2/2)
fn marcum_q_numeric(m: f64, a: f64, b: f64) -> f64 {
  if b <= 0.0 {
    return 1.0;
  }
  let p = a * a / 2.0;
  let y = b * b / 2.0;
  if p == 0.0 {
    return gamma_regularized_numeric(m, y);
  }
  let mut sum = 0.0;
  let mut pois = (-p).exp();
  let mut q = gamma_regularized_numeric(m, y);
  // increment term: y^s e^(-y) / Gamma(s + 1), starting at s = m
  let mut s = m;
  let mut inc = (s * y.ln() - y - lgamma(s + 1.0)).exp();
  for n in 1..=4000 {
    sum += pois * q;
    pois *= p / n as f64;
    q += inc;
    s += 1.0;
    inc *= y / s;
    if pois < 1e-18 && n as f64 > p {
      break;
    }
  }
  sum
}

/// Owen's T function, T(h, a) = (1/2π) ∫₀ᵃ e^(-h²(1+x²)/2)/(1+x²) dx, by
/// composite Simpson quadrature. It is even in h and odd in a.
fn owen_t_numeric(h: f64, a: f64) -> f64 {
  if a == 0.0 {
    return 0.0;
  }
  let a_abs = a.abs();
  let h2 = h * h;
  let f = |x: f64| (-0.5 * h2 * (1.0 + x * x)).exp() / (1.0 + x * x);
  // Even panel count; scale resolution with the interval length.
  let n = (((a_abs * 4000.0).round() as usize).clamp(2000, 1_000_000) / 2) * 2;
  let step = a_abs / n as f64;
  let mut sum = f(0.0) + f(a_abs);
  for i in 1..n {
    let x = i as f64 * step;
    sum += if i.is_multiple_of(2) { 2.0 } else { 4.0 } * f(x);
  }
  let integral = sum * step / 3.0;
  a.signum() * integral / (2.0 * std::f64::consts::PI)
}

/// OwenT[h, a] — Owen's T function. Exact values: T(h, 0) = 0 and, for an exact
/// h = 0, T(0, a) = ArcTan[a]/(2π). Otherwise it evaluates numerically when an
/// argument is inexact, and stays symbolic for exact non-special arguments
/// (matching wolframscript).
pub fn owen_t_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("OwenT", args));
  if args.len() != 2 {
    return unevaluated();
  }
  let (h, a) = (&args[0], &args[1]);
  let has_real = args.iter().any(|e| matches!(e, Expr::Real(_)));

  // T(h, 0) = 0.
  if is_expr_zero(a) {
    return Ok(if has_real {
      Expr::Real(0.0)
    } else {
      Expr::Integer(0)
    });
  }
  // Exact h = 0: T(0, a) = ArcTan[a]/(2π).
  if matches!(h, Expr::Integer(0)) && !has_real {
    return Ok(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::FunctionCall {
        name: "ArcTan".to_string(),
        args: vec![a.clone()].into(),
      }),
      right: Box::new(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(2), Expr::Identifier("Pi".to_string())].into(),
      }),
    });
  }
  // Numeric evaluation requires an inexact argument.
  if has_real
    && let (Some(hv), Some(av)) = (
      crate::functions::math_ast::numeric_utils::try_eval_to_f64(h),
      crate::functions::math_ast::numeric_utils::try_eval_to_f64(a),
    )
  {
    return Ok(Expr::Real(owen_t_numeric(hv, av)));
  }
  unevaluated()
}

/// Compute Q(a, z) = 1 - P(a, z) numerically
pub(crate) fn gamma_regularized_numeric(a: f64, z: f64) -> f64 {
  if z <= 0.0 {
    return 1.0;
  }

  if z < a + 1.0 {
    // Series expansion for P(a, z)
    let ln_prefix = a * z.ln() - z - lgamma(a);
    let prefix = ln_prefix.exp();

    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;
    for n in 1..200 {
      term *= z / (a + n as f64);
      sum += term;
      if term.abs() < 1e-15 * sum.abs() {
        break;
      }
    }
    1.0 - prefix * sum
  } else {
    // Continued fraction for Q(a, z) via modified Lentz:
    // Q = prefix / (z+1-a - 1(1-a)/(z+3-a - 2(2-a)/(z+5-a - ...)))
    let ln_prefix = a * z.ln() - z - lgamma(a);
    let prefix = ln_prefix.exp();

    let fpmin = 1e-300;
    let mut b = z + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / if b.abs() < fpmin { fpmin } else { b };
    let mut h = d;
    for i in 1..200 {
      let an = -(i as f64) * (i as f64 - a);
      b += 2.0;
      d = an * d + b;
      if d.abs() < fpmin {
        d = fpmin;
      }
      c = b + an / c;
      if c.abs() < fpmin {
        c = fpmin;
      }
      d = 1.0 / d;
      let delta = d * c;
      h *= delta;
      if (delta - 1.0).abs() < 1e-15 {
        break;
      }
    }

    prefix * h
  }
}

/// Numeric inverse of the upper regularized incomplete gamma function:
/// returns z such that `gamma_regularized_numeric(a, z) == q`. Because Q(a, z)
/// decreases monotonically from 1 (at z = 0) to 0 (as z -> Infinity), the root
/// is found by bracketing then bisection.
fn inverse_gamma_regularized_numeric(a: f64, q: f64) -> f64 {
  if q <= 0.0 {
    return f64::INFINITY;
  }
  if q >= 1.0 {
    return 0.0;
  }
  let mut lo = 0.0_f64;
  let mut hi = (a + 1.0).max(1.0);
  let mut guard = 0;
  while gamma_regularized_numeric(a, hi) > q && guard < 100 {
    hi *= 2.0;
    guard += 1;
  }
  for _ in 0..200 {
    let mid = 0.5 * (lo + hi);
    if gamma_regularized_numeric(a, mid) > q {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  0.5 * (lo + hi)
}

/// InverseGammaRegularized[a, q] - inverse of the upper regularized incomplete
/// gamma function: returns z with GammaRegularized[a, z] == q. The 3-arg form
/// InverseGammaRegularized[a, z0, q] inverts GammaRegularized[a, z0, z]; for
/// z0 == 0 this reduces to InverseGammaRegularized[a, 1 - q]. Exact inputs stay
/// symbolic (matching wolframscript); inexact (machine-real) inputs evaluate
/// numerically by bisection.
pub fn inverse_gamma_regularized_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let symbolic = || Ok(unevaluated("InverseGammaRegularized", args));

  // 3-arg form InverseGammaRegularized[a, z0, q] solves
  // GammaRegularized[a, z0, z] = Q(a, z0) - Q(a, z) == q for z, i.e.
  // Q(a, z) = Q(a, z0) - q. Exact arguments stay symbolic (matching
  // wolframscript, which keeps even InverseGammaRegularized[2, 0, 1/2] in the
  // 3-arg form); inexact arguments evaluate numerically.
  if args.len() == 3 {
    let inexact = args.iter().any(|a| matches!(a, Expr::Real(_)));
    if let (true, Some(av), Some(z0v), Some(qv)) = (
      inexact,
      expr_to_f64(&args[0]),
      expr_to_f64(&args[1]),
      expr_to_f64(&args[2]),
    ) && av > 0.0
    {
      let target = gamma_regularized_numeric(av, z0v) - qv;
      return Ok(Expr::Real(inverse_gamma_regularized_numeric(av, target)));
    }
    return symbolic();
  }
  if args.len() != 2 {
    return symbolic();
  }

  let a = &args[0];
  let q = &args[1];
  let a_num = expr_to_f64(a);
  let q_num = expr_to_f64(q);

  // Boundary values for positive numeric a: q == 0 -> Infinity, q == 1 -> 0.
  if let (Some(av), Some(qv)) = (a_num, q_num)
    && av > 0.0
  {
    if qv == 0.0 {
      return Ok(Expr::Identifier("Infinity".to_string()));
    }
    if qv == 1.0 {
      return Ok(Expr::Integer(0));
    }
  }

  // Elementary closed form for a == 1: z = -Log[q]. wolframscript applies it
  // when q is a concrete number (e.g. InverseGammaRegularized[1, 1/2] = Log[2],
  // and the machine-Real case), but keeps InverseGammaRegularized[1, q]
  // unevaluated for a free symbol q.
  let q_is_number =
    matches!(q, Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_))
      || matches!(q, Expr::FunctionCall { name, .. } if name == "Rational");
  if matches!(a, Expr::Integer(1)) && q_is_number {
    let neg_log = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(-1)),
      right: Box::new(Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![q.clone()].into(),
      }),
    };
    return crate::evaluator::evaluate_expr_to_expr(&neg_log);
  }

  // Inexact (machine-real) input -> numeric bisection.
  let inexact = matches!(a, Expr::Real(_)) || matches!(q, Expr::Real(_));
  if let (true, Some(av), Some(qv)) = (inexact, a_num, q_num)
    && av > 0.0
  {
    return Ok(Expr::Real(inverse_gamma_regularized_numeric(av, qv)));
  }

  symbolic()
}

/// BarnesG[z] - Barnes G-function.
/// For positive integers n: G(n) = product of factorials = prod_{k=0}^{n-2} k!
/// G(1) = 1, G(n+1) = Gamma(n) * G(n)
/// For non-positive integers: G(n) = 0
/// For real values: numerical computation via log Barnes G
pub fn barnes_g_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "BarnesG expects 1 argument".into(),
    ));
  }

  match &args[0] {
    Expr::Integer(n) => {
      let n = *n;
      if n <= 0 {
        return Ok(Expr::Integer(0));
      }
      if n == 1 {
        return Ok(Expr::Integer(1));
      }
      // G(n) = product_{k=0}^{n-2} k!
      // Use recurrence: G(n+1) = Gamma(n) * G(n) = (n-1)! * G(n)
      // G(1) = 1, G(2) = 1, G(3) = 1, G(4) = 2, G(5) = 12, ...
      let mut result = BigInt::from(1);
      let mut factorial = BigInt::from(1);
      // factorial tracks (k-1)! as we compute G(k+1) = (k-1)! * G(k)
      for k in 1..n - 1 {
        factorial *= BigInt::from(k);
        result *= &factorial;
      }
      // Try to convert to i128
      use num_traits::ToPrimitive;
      match result.to_i128() {
        Some(v) => Ok(Expr::Integer(v)),
        None => Ok(Expr::BigInteger(result)),
      }
    }
    Expr::Real(x) => {
      let val = barnes_g_float(*x);
      Ok(Expr::Real(val))
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "Rational" && fargs.len() == 2 =>
    {
      // For N[BarnesG[rational]], return unevaluated; N[] will handle conversion
      Ok(unevaluated("BarnesG", args))
    }
    _ => Ok(unevaluated("BarnesG", args)),
  }
}

/// Compute Barnes G function for real z numerically.
/// Shifts z to [1, 2] via recurrence, then uses the Taylor series
/// derived from the Weierstrass product.
fn barnes_g_float(z: f64) -> f64 {
  if z == f64::INFINITY {
    return f64::INFINITY;
  }
  if z.is_nan() || z == f64::NEG_INFINITY {
    return f64::NAN;
  }

  // For non-positive integers, return 0
  if z <= 0.0 && z == z.floor() {
    return 0.0;
  }

  log_barnes_g_float(z).exp()
}

/// ln G(z) for z away from the non-positive integers (where G vanishes
/// and the logarithm diverges); shared by BarnesG and LogBarnesG.
fn log_barnes_g_float(z: f64) -> f64 {
  // Shift z to [1, 2] using recurrence G(z+1) = Gamma(z) * G(z)
  let mut z_shifted = z;
  let mut log_correction = 0.0_f64;

  while z_shifted < 1.0 {
    log_correction -= lgamma(z_shifted);
    z_shifted += 1.0;
  }
  while z_shifted > 1.5 {
    z_shifted -= 1.0;
    log_correction += lgamma(z_shifted);
  }

  let w = z_shifted - 1.0; // w in [0, 1]
  log_barnes_g_series(w) + log_correction
}

/// LogBarnesG[z] - logarithm of the Barnes G-function.
/// Positive integers give the exact Log[G(n)] (0 when G(n) == 1);
/// non-positive integers (exact or real-valued) give -Infinity;
/// machine reals evaluate numerically; exact non-integers and symbols
/// stay unevaluated.
///
/// The machine-real path shares BarnesG's series and agrees with
/// wolframscript to ~12 significant digits, not to the last bit
/// (wolframscript uses its own internal asymptotics: e.g. it prints
/// LogBarnesG[5.] as 2.4849066497880052 while Log[12.] is
/// 2.4849066497880004).
pub fn log_barnes_g_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated("LogBarnesG", args);
  if args.len() != 1 {
    return Ok(unevaluated(args));
  }
  let neg_infinity = || Expr::UnaryOp {
    op: UnaryOperator::Minus,
    operand: Box::new(Expr::Identifier("Infinity".to_string())),
  };
  match &args[0] {
    Expr::Integer(n) => {
      if *n <= 0 {
        return Ok(neg_infinity());
      }
      let g = barnes_g_ast(&[args[0].clone()])?;
      if matches!(g, Expr::Integer(1)) {
        return Ok(Expr::Integer(0));
      }
      Ok(Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![g].into(),
      })
    }
    Expr::Real(x) => {
      if *x <= 0.0 && *x == x.floor() {
        return Ok(neg_infinity());
      }
      Ok(Expr::Real(log_barnes_g_float(*x)))
    }
    _ => Ok(unevaluated(args)),
  }
}

/// Taylor series for ln G(1+z), valid for |z| ≤ 1.
/// Derived from the Weierstrass product:
/// ln G(1+z) = z/2 * ln(2π) - [z + (1+γ)z²]/2
///             + Σ_{m=2}^{∞} (-1)^m ζ(m)/(m+1) * z^{m+1}
fn log_barnes_g_series(z: f64) -> f64 {
  if z.abs() < 1e-15 {
    return 0.0;
  }

  let log_2pi = (2.0 * std::f64::consts::PI).ln();
  let gamma_e = 0.5772156649015329;

  let mut result = z / 2.0 * log_2pi - (z + (1.0 + gamma_e) * z * z) / 2.0;

  // Precomputed ζ(m) values for m=2..40
  let zeta: [f64; 39] = [
    1.6449340668482264,    // ζ(2)
    1.2020569031595943,    // ζ(3)
    1.0823232337111382,    // ζ(4)
    1.036_927_755_143_37,  // ζ(5)
    1.017_343_061_984_449, // ζ(6)
    1.0083492773819228,    // ζ(7)
    1.0040773561979443,    // ζ(8)
    1.0020083928260822,    // ζ(9)
    1.0009945751278181,    // ζ(10)
    1.0004941886041195,    // ζ(11)
    1.000_246_086_553_308, // ζ(12)
    1.0001227133475785,    // ζ(13)
    1.0000612481350587,    // ζ(14)
    1.000_030_588_236_307, // ζ(15)
    1.0000152822594086,    // ζ(16)
    1.0000076371976379,    // ζ(17)
    1.000_003_817_293_265, // ζ(18)
    1.0000019082127165,    // ζ(19)
    1.0000009539620339,    // ζ(20)
    1.0000004769329868,    // ζ(21)
    1.0000002384505027,    // ζ(22)
    1.000_000_119_219_926, // ζ(23)
    1.000_000_059_608_189, // ζ(24)
    1.0000000298035035,    // ζ(25)
    1.0000000149015548,    // ζ(26)
    1.0000000074507118,    // ζ(27)
    1.000_000_003_725_334, // ζ(28)
    1.0000000018626598,    // ζ(29)
    1.0000000009313274,    // ζ(30)
    1.0000000004656629,    // ζ(31)
    1.0000000002328312,    // ζ(32)
    1.0000000001164155,    // ζ(33)
    1.0000000000582077,    // ζ(34)
    1.0000000000291039,    // ζ(35)
    1.000_000_000_014_552, // ζ(36)
    1.000_000_000_007_276, // ζ(37)
    1.000_000_000_003_638, // ζ(38)
    1.000_000_000_001_819, // ζ(39)
    1.0000000000009095,    // ζ(40)
  ];

  let mut z_power = z * z * z; // z^3 (m=2 gives z^{m+1} = z^3)
  for (i, &zeta_m) in zeta.iter().enumerate() {
    let m = (i + 2) as f64;
    let sign = if (i + 2) % 2 == 0 { 1.0 } else { -1.0 };
    let term = sign * zeta_m / (m + 1.0) * z_power;
    result += term;
    if term.abs() < 1e-17 {
      break;
    }
    z_power *= z;
  }

  result
}
