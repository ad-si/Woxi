#[allow(unused_imports)]
use super::*;

/// FunctionExpand[expr] — expand special mathematical functions into simpler forms.
pub fn function_expand_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "FunctionExpand expects 1 argument".into(),
    ));
  }
  let expr = &args[0];
  let result = function_expand_inner(expr)?;
  // Evaluate the result to simplify
  crate::evaluator::evaluate_expr_to_expr(&result)
}

fn function_expand_inner(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::FunctionCall { name, args } => {
      // First recursively expand arguments
      let expanded_args: Vec<Expr> = args
        .iter()
        .map(function_expand_inner)
        .collect::<Result<Vec<_>, _>>()?;

      // Then try to expand this function call
      if let Some(expanded) = try_expand_function(name, &expanded_args) {
        return Ok(expanded);
      }

      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: expanded_args.into(),
      })
    }
    Expr::List(items) => {
      let expanded: Vec<Expr> = items
        .iter()
        .map(function_expand_inner)
        .collect::<Result<Vec<_>, _>>()?;
      Ok(Expr::List(expanded.into()))
    }
    Expr::BinaryOp { op, left, right } => {
      let l = function_expand_inner(left)?;
      let r = function_expand_inner(right)?;
      // A `base^exp` written with the Power operator is routed through the
      // same expansion rules as the Power[...] head (e.g. Abs[z]^2).
      if matches!(op, BinaryOperator::Power)
        && let Some(expanded) =
          try_expand_function("Power", &[l.clone(), r.clone()])
      {
        return Ok(expanded);
      }
      Ok(Expr::BinaryOp {
        op: *op,
        left: Box::new(l),
        right: Box::new(r),
      })
    }
    Expr::UnaryOp { op, operand } => {
      let o = function_expand_inner(operand)?;
      Ok(Expr::UnaryOp {
        op: *op,
        operand: Box::new(o),
      })
    }
    _ => Ok(expr.clone()),
  }
}

fn mk_id(name: &str) -> Expr {
  Expr::Identifier(name.to_string())
}

fn mk_plus(a: Expr, b: Expr) -> Expr {
  call("Plus", vec![a, b])
}

/// The additive terms of `e` if its head is `Plus` (either the `Plus[…]` call
/// form or the `+` binary-operator form); otherwise None.
fn as_plus_terms(e: &Expr) -> Option<Vec<Expr>> {
  match e {
    Expr::FunctionCall { name, args } if name == "Plus" => Some(args.to_vec()),
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => Some(vec![(**left).clone(), (**right).clone()]),
    // `a - b` is the same sum with the second term negated.
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => Some(vec![
      (**left).clone(),
      mk_times(mk_int(-1), (**right).clone()),
    ]),
    _ => None,
  }
}

fn mk_div(a: Expr, b: Expr) -> Expr {
  mk_times(a, mk_power(b, mk_int(-1)))
}

/// Whether `e` is `ArcSin[x]` / `ArcCos[x]` / … applied to a single argument.
fn is_inverse_trig_call(e: &Expr) -> bool {
  matches!(e, Expr::FunctionCall { name, args }
  if args.len() == 1
    && matches!(
      name.as_str(),
      "ArcSin" | "ArcCos" | "ArcTan" | "ArcCot" | "ArcSec" | "ArcCsc"
    ))
}

/// Whether `arg` is an inverse-trig call, optionally with an integer multiplier
/// (`ArcSin[x]`, `2 ArcSin[x]`, `3 ArcCos[x]`, …).
fn is_multiple_of_inverse_trig(arg: &Expr) -> bool {
  if is_inverse_trig_call(arg) {
    return true;
  }
  let is_int_times = |a: &Expr, b: &Expr| {
    (matches!(a, Expr::Integer(_)) && is_inverse_trig_call(b))
      || (matches!(b, Expr::Integer(_)) && is_inverse_trig_call(a))
  };
  match arg {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => is_int_times(left, right),
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      is_int_times(&args[0], &args[1])
    }
    _ => false,
  }
}

/// Whether `e` is a polynomial with non-negative integer powers and numeric
/// coefficients — i.e. free of Sqrt, negative/fractional powers, division, and
/// any residual (inverse-)trig heads.
fn is_clean_polynomial(e: &Expr) -> bool {
  match e {
    Expr::Integer(_)
    | Expr::BigInteger(_)
    | Expr::Real(_)
    | Expr::Identifier(_) => true,
    Expr::BinaryOp { op, left, right } => match op {
      BinaryOperator::Plus | BinaryOperator::Minus | BinaryOperator::Times => {
        is_clean_polynomial(left) && is_clean_polynomial(right)
      }
      BinaryOperator::Power => {
        is_clean_polynomial(left)
          && matches!(right.as_ref(), Expr::Integer(n) if *n >= 0)
      }
      _ => false,
    },
    Expr::UnaryOp { operand, .. } => is_clean_polynomial(operand),
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Plus" | "Times" => args.iter().all(is_clean_polynomial),
      "Rational" => true,
      "Power" if args.len() == 2 => {
        is_clean_polynomial(&args[0])
          && matches!(&args[1], Expr::Integer(n) if *n >= 0)
      }
      _ => false,
    },
    _ => false,
  }
}

/// The negation of `t` when `t` is a negative term — a unary minus, or a
/// product with a negative numeric leading (or trailing) coefficient. Returns
/// None for a term that is not negative. Both the `Times[…]` call form and the
/// `*` binary-operator form are recognised, since `Expand` produces either.
fn negate_if_negative(t: &Expr) -> Option<Expr> {
  // `Rational[-p, q]` → `Rational[p, q]`, the coefficient form `-x^2/2` takes.
  let negate_ratio = |e: &Expr| -> Option<Expr> {
    match e {
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        match (&args[0], &args[1]) {
          (Expr::Integer(p), Expr::Integer(q)) if *p < 0 => {
            Some(mk_ratio(-p, *q))
          }
          _ => None,
        }
      }
      _ => None,
    }
  };
  let strip_neg_coeff = |coeff: &Expr, rest: Vec<Expr>| -> Option<Expr> {
    let positive = match coeff {
      Expr::Integer(-1) => None,
      Expr::Integer(n) if *n < 0 => Some(mk_int(-n)),
      Expr::Real(r) if *r < 0.0 => Some(Expr::Real(-r)),
      _ => Some(negate_ratio(coeff)?),
    };
    let mut factors: Vec<Expr> = positive.into_iter().collect();
    factors.extend(rest);
    Some(match factors.len() {
      0 => mk_int(1),
      1 => factors.into_iter().next().unwrap(),
      _ => call("Times", factors),
    })
  };
  match t {
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => Some((**operand).clone()),
    Expr::Integer(n) if *n < 0 => Some(mk_int(-n)),
    Expr::Real(r) if *r < 0.0 => Some(Expr::Real(-r)),
    Expr::FunctionCall { name, .. } if name == "Rational" => negate_ratio(t),
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      strip_neg_coeff(&args[0], args[1..].to_vec()).or_else(|| {
        strip_neg_coeff(&args[args.len() - 1], args[..args.len() - 1].to_vec())
      })
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => strip_neg_coeff(left, vec![(**right).clone()])
      .or_else(|| strip_neg_coeff(right, vec![(**left).clone()])),
    _ => None,
  }
}

/// A positive rational as a normalised `(numerator, denominator)` pair.
type Ratio = (i128, i128);

fn gcd_i128(a: i128, b: i128) -> i128 {
  if b == 0 { a.abs() } else { gcd_i128(b, a % b) }
}

fn ratio(n: i128, d: i128) -> Ratio {
  let g = gcd_i128(n, d).max(1);
  (n / g, d / g)
}

fn ratio_times(a: Ratio, b: Ratio) -> Ratio {
  ratio(a.0 * b.0, a.1 * b.1)
}

fn ratio_over(a: Ratio, b: Ratio) -> Ratio {
  ratio(a.0 * b.1, a.1 * b.0)
}

/// The exact square root of a non-negative integer, if it has one.
fn integer_sqrt(n: i128) -> Option<i128> {
  if n < 0 {
    return None;
  }
  let mut r = (n as f64).sqrt().round() as i128;
  // The float round-trip can be off by one for large inputs.
  while r > 0 && r * r > n {
    r -= 1;
  }
  while (r + 1) * (r + 1) <= n {
    r += 1;
  }
  (r * r == n).then_some(r)
}

/// The exact square root of a positive rational, if it has one.
fn ratio_sqrt(r: Ratio) -> Option<Ratio> {
  Some((integer_sqrt(r.0)?, integer_sqrt(r.1)?))
}

/// The squarefree kernel of a positive integer: the smallest `k` with `n * k`
/// a perfect square.
fn squarefree_kernel(mut n: i128) -> i128 {
  let mut kernel = 1;
  let mut factor = 2;
  while factor * factor <= n {
    let mut multiplicity = 0;
    while n % factor == 0 {
      n /= factor;
      multiplicity += 1;
    }
    if multiplicity % 2 == 1 {
      kernel *= factor;
    }
    factor += 1;
  }
  kernel * n
}

/// The positive rational value of `e`, when it is one written exactly.
fn as_positive_ratio(e: &Expr) -> Option<Ratio> {
  match e {
    Expr::Integer(n) if *n > 0 => Some((*n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(p), Expr::Integer(q)) if *p > 0 && *q > 0 => {
          Some(ratio(*p, *q))
        }
        _ => None,
      }
    }
    _ => None,
  }
}

/// The multiplicative factors of `e`, flattening `Times[…]` and `a * b`.
fn as_times_factors(e: &Expr) -> Vec<Expr> {
  match e {
    Expr::FunctionCall { name, args } if name == "Times" => args.to_vec(),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => vec![(**left).clone(), (**right).clone()],
    _ => vec![e.clone()],
  }
}

/// `u^2` or `u^4` (whichever it is), as `(exponent, u)`.
fn as_even_power(e: &Expr) -> Option<(i128, Expr)> {
  let (base, exponent) = match e {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => ((**left).clone(), (**right).clone()),
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      (args[0].clone(), args[1].clone())
    }
    _ => return None,
  };
  match exponent {
    // wolframscript only splits the radicand of a quadratic or a quartic;
    // `Sqrt[1 - x^6]` and higher stay as written.
    Expr::Integer(n @ (2 | 4)) => Some((n, base)),
    _ => None,
  }
}

/// A negated term of the form `c u^2` or `c u^4`, split into the positive
/// rational `c` and the square root `u` (or `u^2`) of the power.
fn as_scaled_square(e: &Expr) -> Option<(Ratio, Expr)> {
  let mut coefficient: Ratio = (1, 1);
  let mut power: Option<(i128, Expr)> = None;
  for factor in as_times_factors(e) {
    if let Some(r) = as_positive_ratio(&factor) {
      coefficient = ratio_times(coefficient, r);
    } else if power.is_none() {
      power = Some(as_even_power(&factor)?);
    } else {
      // More than one symbolic factor (`x^2 y^2`): not a form wolframscript
      // splits.
      return None;
    }
  }
  let (exponent, base) = power?;
  let root = if exponent == 2 {
    base
  } else {
    mk_power(base, mk_int(exponent / 2))
  };
  Some((coefficient, root))
}

/// Whether `e` is an exact positive constant — a rational, `Pi`, `Sqrt[2]`, …
/// The `a^2` side of the split has to be one: wolframscript leaves
/// `Sqrt[x^2 - 1]` and `Sqrt[x^2 - y^2]` alone, where the sign of the leading
/// term is unknown.
fn is_exact_positive_constant(e: &Expr) -> bool {
  if !crate::functions::predicate_ast::is_numeric_q(e) || contains_real(e) {
    return false;
  }
  let value = call1("N", e.clone());
  matches!(
    crate::evaluator::evaluate_expr_to_expr(&value),
    Ok(Expr::Real(r)) if r > 0.0
  )
}

/// Whether `e` contains a machine or arbitrary-precision real anywhere.
fn contains_real(e: &Expr) -> bool {
  match e {
    Expr::Real(_) | Expr::BigFloat(_, _) => true,
    Expr::FunctionCall { args, .. } => args.iter().any(contains_real),
    Expr::BinaryOp { left, right, .. } => {
      contains_real(left) || contains_real(right)
    }
    Expr::UnaryOp { operand, .. } => contains_real(operand),
    _ => false,
  }
}

/// `Sqrt[a - c u^2]` → `Sqrt[a'] (Sqrt[a'' - b] Sqrt[a'' + b])`, applied
/// recursively to the first factor so `Sqrt[1 - x^4]` becomes
/// `Sqrt[1-x] Sqrt[1+x] Sqrt[1+x^2]`. This is the form `FunctionExpand`
/// produces, since it assumes nothing about the sign of the parts.
///
/// The positive term must be an exact positive constant and the negated one a
/// rational multiple of a square or fourth power. When neither coefficient is
/// a perfect square the radicand is scaled by the squarefree kernel of the
/// negated coefficient so that one becomes a square, and the compensating
/// factor stays outside — `Sqrt[1 - 2 x^2]` gives
/// `Sqrt[Sqrt[2] - 2 x] Sqrt[Sqrt[2] + 2 x] / Sqrt[2]`, as wolframscript
/// prints it. Returns None for a radicand of any other shape.
fn split_sqrt_of_square_difference(radicand: &Expr) -> Option<Expr> {
  let terms = as_plus_terms(radicand)?;
  if terms.len() != 2 {
    return None;
  }
  // Exactly one term must be negated; it supplies b^2, the other a^2.
  let negated = |t: &Expr| -> Option<Expr> { negate_if_negative(t) };
  let (a_sq, b_sq) = match (negated(&terms[0]), negated(&terms[1])) {
    (None, Some(b)) => (terms[0].clone(), b),
    (Some(b), None) => (terms[1].clone(), b),
    _ => return None,
  };
  if !is_exact_positive_constant(&a_sq) {
    return None;
  }
  let (cb, base) = as_scaled_square(&b_sq)?;
  let ca = as_positive_ratio(&a_sq);

  // `prefactor` is pulled out of the radical as `Sqrt[prefactor]`; `a` and
  // `b` are the two sides of the rescaled difference of squares.
  let (prefactor, a, b): (Ratio, Expr, Expr) = if let Some(ca) = ca
    && ca != (1, 1)
    && ratio_over(cb, ca).1 == 1
    && integer_sqrt(ratio_over(cb, ca).0).is_some()
  {
    // The whole constant divides out and leaves a perfect square behind:
    // `Sqrt[3 - 12 x^2]` → `Sqrt[3] Sqrt[1 - 2 x] Sqrt[1 + 2 x]`.
    let scale = integer_sqrt(ratio_over(cb, ca).0)?;
    (ca, mk_int(1), mk_times(mk_int(scale), base))
  } else if let Some(ca) = ca
    && let (Some(ra), Some(rb)) = (ratio_sqrt(ca), ratio_sqrt(cb))
  {
    // Both coefficients are already squares: `Sqrt[4 - 9 x^2]` →
    // `Sqrt[2 - 3 x] Sqrt[2 + 3 x]`.
    (
      (1, 1),
      mk_exact_ratio(ra),
      mk_times(mk_exact_ratio(rb), base),
    )
  } else {
    // Scale the radicand so the negated coefficient becomes a square, and
    // divide the result by the square root of that scale.
    let scale = squarefree_kernel(cb.0 * cb.1);
    let scaled_b = ratio_sqrt(ratio_times(cb, (scale, 1)))?;
    let scaled_a = match ca {
      Some(ca) => call1("Sqrt", mk_exact_ratio(ratio_times(ca, (scale, 1)))),
      None if scale == 1 => call1("Sqrt", a_sq.clone()),
      None => call1("Sqrt", mk_times(mk_int(scale), a_sq.clone())),
    };
    (
      (1, scale),
      scaled_a,
      mk_times(mk_exact_ratio(scaled_b), base),
    )
  };

  let minus = mk_plus(a.clone(), mk_times(mk_int(-1), b.clone()));
  let plus = mk_plus(a, b);
  // Each side is expanded, so `Sqrt[9 - 4 (x + 1)^2]` prints as
  // `Sqrt[1 - 2 x] Sqrt[5 + 2 x]` rather than keeping the shifted square.
  let first = split_sqrt_of_square_difference(&minus)
    .unwrap_or_else(|| call1("Sqrt", call1("Expand", minus)));
  let split = mk_times(first, call1("Sqrt", call1("Expand", plus)));
  Some(if prefactor == (1, 1) {
    split
  } else {
    mk_times(call1("Sqrt", mk_exact_ratio(prefactor)), split)
  })
}

/// A positive rational as an `Integer` or `Rational[…]` expression.
fn mk_exact_ratio(r: Ratio) -> Expr {
  if r.1 == 1 {
    mk_int(r.0)
  } else {
    mk_ratio(r.0, r.1)
  }
}

/// The radicand of `e` when `e` is `Sqrt[r]` or `r^(1/2)`.
fn as_sqrt_radicand(e: &Expr) -> Option<&Expr> {
  let is_half = |x: &Expr| {
    matches!(x, Expr::FunctionCall { name, args }
      if name == "Rational"
        && args.len() == 2
        && matches!(&args[0], Expr::Integer(1))
        && matches!(&args[1], Expr::Integer(2)))
  };
  match e {
    Expr::FunctionCall { name, args } if name == "Sqrt" && args.len() == 1 => {
      Some(&args[0])
    }
    Expr::FunctionCall { name, args }
      if name == "Power" && args.len() == 2 && is_half(&args[1]) =>
    {
      Some(&args[0])
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } if is_half(right) => Some(left.as_ref()),
    _ => None,
  }
}

/// Try to expand a specific function call. Returns None if no expansion applies.
fn try_expand_function(name: &str, args: &[Expr]) -> Option<Expr> {
  // Trig of an integer multiple of an inverse trig function (e.g.
  // Cos[2 ArcSin[x]] = 1 - 2 x^2) expands via multiple-angle identities.
  // Route through TrigExpand + Expand; a clean polynomial is the answer, and
  // otherwise the residual `Sqrt[1 - x^2]` radicals are split the way Wolfram
  // does (Sin[2 ArcSin[x]] = 2 Sqrt[1 - x] x Sqrt[1 + x]).
  if matches!(name, "Sin" | "Cos" | "Tan" | "Cot" | "Sec" | "Csc")
    && args.len() == 1
    && is_multiple_of_inverse_trig(&args[0])
  {
    let trig = call1("TrigExpand", call1(name, args[0].clone()));
    let expanded = call1("Expand", trig);
    if let Ok(result) = crate::evaluator::evaluate_expr_to_expr(&expanded) {
      if is_clean_polynomial(&result) {
        return Some(result);
      }
      if let Ok(split) = function_expand_inner(&result)
        && crate::syntax::expr_to_string(&split)
          != crate::syntax::expr_to_string(&result)
        && let Ok(v) = crate::evaluator::evaluate_expr_to_expr(&split)
      {
        return Some(v);
      }
    }
  }

  // FunctionExpand[Sqrt[a^2 - b^2]] = Sqrt[a - b] Sqrt[a + b].
  {
    let call = call(name, args.to_vec());
    if let Some(radicand) = as_sqrt_radicand(&call)
      && let Some(split) = split_sqrt_of_square_difference(radicand)
    {
      return Some(split);
    }
  }

  match name {
    // HurwitzZeta[m, a] with an integer m >= 2 → (-1)^m/(m-1)! PolyGamma[m-1, a].
    "HurwitzZeta" if args.len() == 2 => {
      if let Expr::Integer(m) = &args[0]
        && *m >= 2
      {
        let coeff = mk_div(
          mk_power(mk_int(-1), mk_int(*m)),
          call("Factorial", vec![mk_int(*m - 1)]),
        );
        Some(mk_times(
          coeff,
          call("PolyGamma", vec![mk_int(*m - 1), args[1].clone()]),
        ))
      } else {
        None
      }
    }

    // Pochhammer[a, n] → Gamma[a + n] / Gamma[a]
    "Pochhammer" if args.len() == 2 => {
      let a = &args[0];
      let n = &args[1];
      Some(mk_div(
        call1("Gamma", mk_plus(a.clone(), n.clone())),
        call1("Gamma", a.clone()),
      ))
    }

    // FactorialPower[x, n] = x (x-1) … (x-n+1); the step-h form
    // FactorialPower[x, n, h] = x (x-h) … (x-(n-1)h). A non-negative integer n
    // expands to the explicit product; a symbolic n gives the Gamma ratio
    // Gamma[1+x]/Gamma[1-n+x].
    "FactorialPower" if args.len() == 2 || args.len() == 3 => {
      let x = &args[0];
      let n = &args[1];
      let h = if args.len() == 3 {
        Some(&args[2])
      } else {
        None
      };
      match n {
        // Product_{k=0}^{n-1} (x - k h), with h defaulting to 1. Only handled
        // for a numeric (or absent) step h; a symbolic step is left to the
        // caller since wolframscript's factored form uses a different Times
        // ordering that Woxi's canonicalizer does not reproduce.
        Expr::Integer(nn)
          if *nn >= 0
            && h.is_none_or(|e| {
              crate::functions::predicate_ast::is_numeric_q(e)
            }) =>
        {
          let nn = *nn as usize;
          if nn == 0 {
            return Some(mk_int(1));
          }
          let hexpr = h.cloned().unwrap_or_else(|| mk_int(1));
          let factors: Vec<Expr> = (0..nn)
            .map(|k| {
              if k == 0 {
                x.clone()
              } else {
                mk_plus(
                  x.clone(),
                  mk_times(mk_int(-(k as i128)), hexpr.clone()),
                )
              }
            })
            .collect();
          Some(factors.into_iter().reduce(mk_times).unwrap_or(mk_int(1)))
        }
        // Symbolic n (2-argument form only): the Gamma-function ratio.
        _ if h.is_none() => Some(mk_div(
          call("Gamma", vec![mk_plus(mk_int(1), x.clone())]),
          call(
            "Gamma",
            vec![call(
              "Plus",
              vec![mk_int(1), mk_times(mk_int(-1), n.clone()), x.clone()],
            )],
          ),
        )),
        _ => None,
      }
    }

    // Beta[a, b] → Gamma[a] * Gamma[b] / Gamma[a + b]
    "Beta" if args.len() == 2 => {
      let a = &args[0];
      let b = &args[1];
      Some(mk_div(
        mk_times(call1("Gamma", a.clone()), call1("Gamma", b.clone())),
        call("Gamma", vec![mk_plus(a.clone(), b.clone())]),
      ))
    }

    // Factorial[n] (i.e. n!) → Gamma[1 + n]
    "Factorial" if args.len() == 1 => {
      Some(call("Gamma", vec![mk_plus(mk_int(1), args[0].clone())]))
    }

    // Abs[z]^(2m) → (Re[z]^2 + Im[z]^2)^m (the squared-modulus identity). Only
    // even integer exponents expand; odd powers keep the Abs.
    "Power"
      if args.len() == 2
        && matches!(&args[0], Expr::FunctionCall { name, args: a }
          if name == "Abs" && a.len() == 1)
        && matches!(&args[1], Expr::Integer(e) if *e > 0 && *e % 2 == 0) =>
    {
      let Expr::FunctionCall { args: a, .. } = &args[0] else {
        unreachable!("guarded above");
      };
      let Expr::Integer(e) = &args[1] else {
        unreachable!("guarded above");
      };
      let z = &a[0];
      let sq_sum = mk_plus(
        mk_power(call1("Re", z.clone()), mk_int(2)),
        mk_power(call1("Im", z.clone()), mk_int(2)),
      );
      let m = e / 2;
      let result = if m == 1 {
        sq_sum
      } else {
        mk_power(sq_sum, mk_int(m))
      };
      // Re[z]/Im[z] of a sum distribute, so expand the freshly built form.
      Some(function_expand_inner(&result).unwrap_or(result))
    }

    // Re[z] / Im[z] distribute over a sum: Re[a + b] → Re[a] + Re[b].
    "Re" | "Im" if args.len() == 1 => {
      let terms = as_plus_terms(&args[0])?;
      Some(call(
        "Plus",
        terms.into_iter().map(|t| call1(name, t)).collect(),
      ))
    }

    // Multinomial[a1, …, ak] = (a1 + … + ak)! / (a1! ⋯ ak!) →
    //   Gamma[1 + a1 + … + ak] / (Gamma[1 + a1] ⋯ Gamma[1 + ak]).
    "Multinomial" if !args.is_empty() => {
      let sum = call("Plus", args.to_vec());
      let numerator = call("Gamma", vec![mk_plus(mk_int(1), sum)]);
      let denominator = args
        .iter()
        .map(|a| call("Gamma", vec![mk_plus(mk_int(1), a.clone())]))
        .reduce(mk_times)
        .unwrap_or_else(|| mk_int(1));
      Some(mk_div(numerator, denominator))
    }

    // Binomial[n, k]: a specific integer k expands to a polynomial; an
    // otherwise symbolic k expands to the Gamma-function form
    //   Gamma[1 + n] / (Gamma[1 + k] * Gamma[1 - k + n]).
    "Binomial" if args.len() == 2 => {
      if let Expr::Integer(k) = &args[1] {
        expand_binomial_integer_k(&args[0], *k)
      } else {
        let n = &args[0];
        let k = &args[1];
        Some(mk_div(
          call("Gamma", vec![mk_plus(mk_int(1), n.clone())]),
          mk_times(
            call("Gamma", vec![mk_plus(mk_int(1), k.clone())]),
            call(
              "Gamma",
              vec![call(
                "Plus",
                vec![mk_int(1), mk_times(mk_int(-1), k.clone()), n.clone()],
              )],
            ),
          ),
        ))
      }
    }

    // CatalanNumber[n] → (2^(2 n) Gamma[1/2 + n]) / (Sqrt[Pi] Gamma[2 + n])
    "CatalanNumber" if args.len() == 1 => {
      let n = &args[0];
      Some(mk_div(
        mk_times(
          mk_power(mk_int(2), mk_times(mk_int(2), n.clone())),
          call1("Gamma", mk_plus(mk_ratio(1, 2), n.clone())),
        ),
        mk_times(
          call1("Sqrt", mk_id("Pi")),
          call1("Gamma", mk_plus(mk_int(2), n.clone())),
        ),
      ))
    }

    // Subfactorial[n] → Gamma[1 + n, -1] / E
    "Subfactorial" if args.len() == 1 => {
      let n = &args[0];
      Some(mk_div(
        call("Gamma", vec![mk_plus(mk_int(1), n.clone()), mk_int(-1)]),
        mk_id("E"),
      ))
    }

    // Haversine[x] → (1 - Cos[x]) / 2
    "Haversine" if args.len() == 1 => Some(mk_times(
      mk_ratio(1, 2),
      mk_plus(
        mk_int(1),
        mk_times(mk_int(-1), call1("Cos", args[0].clone())),
      ),
    )),

    // InverseHaversine[x] → 2 * ArcSin[Sqrt[x]]
    "InverseHaversine" if args.len() == 1 => Some(mk_times(
      mk_int(2),
      call("ArcSin", vec![call1("Sqrt", args[0].clone())]),
    )),

    // InverseGudermannian[x] → Log[Tan[Pi/4 + x/2]]
    "InverseGudermannian" if args.len() == 1 => Some(call(
      "Log",
      vec![call(
        "Tan",
        vec![mk_plus(
          mk_times(mk_ratio(1, 4), mk_id("Pi")),
          mk_times(mk_ratio(1, 2), args[0].clone()),
        )],
      )],
    )),

    // Sinc[x] → Sin[x] / x
    "Sinc" if args.len() == 1 => {
      Some(mk_div(call1("Sin", args[0].clone()), args[0].clone()))
    }

    // LogisticSigmoid[x] → 1/(1 + E^(-x)), i.e. (1 + E^(-x))^(-1).
    "LogisticSigmoid" if args.len() == 1 => Some(mk_power(
      mk_plus(
        mk_int(1),
        mk_power(mk_id("E"), mk_times(mk_int(-1), args[0].clone())),
      ),
      mk_int(-1),
    )),

    // ChebyshevT[n, x] → Cos[n * ArcCos[x]]
    "ChebyshevT" if args.len() == 2 => Some(call(
      "Cos",
      vec![mk_times(args[0].clone(), call1("ArcCos", args[1].clone()))],
    )),

    // ChebyshevU[n, x] → Sin[(1 + n) * ArcCos[x]] / (Sqrt[1 - x] * Sqrt[1 + x])
    "ChebyshevU" if args.len() == 2 => {
      let n = &args[0];
      let x = &args[1];
      Some(mk_div(
        call(
          "Sin",
          vec![mk_times(
            mk_plus(mk_int(1), n.clone()),
            call1("ArcCos", x.clone()),
          )],
        ),
        mk_times(
          call1("Sqrt", mk_plus(mk_int(1), mk_times(mk_int(-1), x.clone()))),
          call1("Sqrt", mk_plus(mk_int(1), x.clone())),
        ),
      ))
    }

    // Fibonacci[n] → (GoldenRatio^n - (-1/GoldenRatio)^n * Cos[n*Pi]) / Sqrt[5]
    // where GoldenRatio = (1 + Sqrt[5]) / 2
    "Fibonacci" if args.len() == 1 => {
      let n = &args[0];
      let sqrt5 = call1("Sqrt", mk_int(5));
      let golden = mk_times(mk_ratio(1, 2), mk_plus(mk_int(1), sqrt5.clone()));
      let inv_golden = mk_times(
        mk_int(2),
        mk_power(mk_plus(mk_int(1), sqrt5.clone()), mk_int(-1)),
      );
      Some(mk_div(
        mk_plus(
          mk_power(golden, n.clone()),
          mk_times(
            mk_int(-1),
            mk_times(
              mk_power(inv_golden, n.clone()),
              call("Cos", vec![mk_times(n.clone(), mk_id("Pi"))]),
            ),
          ),
        ),
        sqrt5,
      ))
    }

    // LucasL[n] → GoldenRatio^n + (-1/GoldenRatio)^n * Cos[n*Pi]
    "LucasL" if args.len() == 1 => {
      let n = &args[0];
      let sqrt5 = call1("Sqrt", mk_int(5));
      let golden = mk_times(mk_ratio(1, 2), mk_plus(mk_int(1), sqrt5.clone()));
      let inv_golden =
        mk_times(mk_int(2), mk_power(mk_plus(mk_int(1), sqrt5), mk_int(-1)));
      Some(mk_plus(
        mk_power(golden, n.clone()),
        mk_times(
          mk_power(inv_golden, n.clone()),
          call("Cos", vec![mk_times(n.clone(), mk_id("Pi"))]),
        ),
      ))
    }

    // Gamma[1/2] → Sqrt[Pi]
    "Gamma" if args.len() == 1 => {
      if let Expr::FunctionCall { name: rn, args: ra } = &args[0]
        && rn == "Rational"
        && ra.len() == 2
        && let (Expr::Integer(1), Expr::Integer(2)) = (&ra[0], &ra[1])
      {
        return Some(call1("Sqrt", mk_id("Pi")));
      }
      None
    }

    // HarmonicNumber[n] -> EulerGamma + PolyGamma[0, 1 + n].
    "HarmonicNumber" if args.len() == 1 => Some(mk_plus(
      mk_id("EulerGamma"),
      call(
        "PolyGamma",
        vec![mk_int(0), mk_plus(mk_int(1), args[0].clone())],
      ),
    )),

    // HarmonicNumber[n, r] -> Zeta[r] - HurwitzZeta[r, 1 + n]. For an integer
    // order r >= 2 the HurwitzZeta is itself reduced to a PolyGamma (reusing the
    // HurwitzZeta rule above), giving e.g. Pi^2/6 - PolyGamma[1, 1 + n].
    "HarmonicNumber" if args.len() == 2 => {
      let r = &args[1];
      let arg = mk_plus(mk_int(1), args[0].clone());
      let hurwitz =
        try_expand_function("HurwitzZeta", &[r.clone(), arg.clone()])
          .unwrap_or_else(|| call("HurwitzZeta", vec![r.clone(), arg]));
      Some(mk_plus(
        call1("Zeta", r.clone()),
        mk_times(mk_int(-1), hurwitz),
      ))
    }

    // Gamma[A]/Gamma[B] with A - B a positive integer expands to the rising
    // factorial Pochhammer[B, A - B] (e.g. Gamma[n+2]/Gamma[n] -> n*(1 + n)).
    "Times" => try_gamma_ratio_in_times(args),

    _ => None,
  }
}

/// If `f` is `Gamma[B]^(-1)` (in either the FunctionCall or BinaryOp Power
/// spelling), return the argument `B`.
fn reciprocal_gamma_arg(f: &Expr) -> Option<Expr> {
  let (base, exp) = match f {
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      (&args[0], &args[1])
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => (left.as_ref(), right.as_ref()),
    _ => return None,
  };
  if !matches!(exp, Expr::Integer(-1)) {
    return None;
  }
  match base {
    Expr::FunctionCall { name, args } if name == "Gamma" && args.len() == 1 => {
      Some(args[0].clone())
    }
    _ => None,
  }
}

/// In a product of `factors`, cancel a `Gamma[A] * Gamma[B]^(-1)` pair whose
/// arguments differ by a positive integer `k = A - B`, replacing it with
/// `Pochhammer[B, k]` (which evaluates to the product B (B+1) ... (B+k-1)).
fn try_gamma_ratio_in_times(factors: &[Expr]) -> Option<Expr> {
  let mut num: Option<(usize, Expr)> = None;
  let mut den: Option<(usize, Expr)> = None;
  for (i, f) in factors.iter().enumerate() {
    if num.is_none()
      && let Expr::FunctionCall { name, args } = f
      && name == "Gamma"
      && args.len() == 1
    {
      num = Some((i, args[0].clone()));
    } else if den.is_none()
      && let Some(b) = reciprocal_gamma_arg(f)
    {
      den = Some((i, b));
    }
  }
  let (ni, a) = num?;
  let (di, b) = den?;
  // k = A - B must be an integer.
  let diff = crate::evaluator::evaluate_expr_to_expr(&mk_plus(
    a.clone(),
    mk_times(mk_int(-1), b.clone()),
  ))
  .ok()?;
  let Expr::Integer(k) = diff else { return None };
  // Gamma[A]/Gamma[B] = Pochhammer[B, k] for k > 0, 1/Pochhammer[A, -k] for
  // k < 0, and 1 for k == 0.
  let poch = match k.cmp(&0) {
    std::cmp::Ordering::Greater => call("Pochhammer", vec![b, mk_int(k)]),
    std::cmp::Ordering::Less => {
      mk_power(call("Pochhammer", vec![a, mk_int(-k)]), mk_int(-1))
    }
    std::cmp::Ordering::Equal => mk_int(1),
  };
  let mut rest: Vec<Expr> = factors
    .iter()
    .enumerate()
    .filter(|(i, _)| *i != ni && *i != di)
    .map(|(_, f)| f.clone())
    .collect();
  rest.push(poch);
  Some(if rest.len() == 1 {
    rest.remove(0)
  } else {
    call("Times", rest)
  })
}

/// Expand Binomial[n, k] for specific small integer k values.
fn expand_binomial_integer_k(n: &Expr, k: i128) -> Option<Expr> {
  match k {
    0 => Some(mk_int(1)),
    1 => Some(n.clone()),
    2 => {
      // n*(n-1)/2
      Some(mk_times(
        mk_ratio(1, 2),
        mk_times(n.clone(), mk_plus(mk_int(-1), n.clone())),
      ))
    }
    3 => {
      // n*(n-1)*(n-2)/6
      Some(mk_times(
        mk_ratio(1, 6),
        mk_times(
          n.clone(),
          mk_times(
            mk_plus(mk_int(-1), n.clone()),
            mk_plus(mk_int(-2), n.clone()),
          ),
        ),
      ))
    }
    _ => None,
  }
}
