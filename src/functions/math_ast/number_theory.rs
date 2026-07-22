#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{
  BinaryOperator, ComparisonOp, Expr, UnaryOperator, bool_expr, expr_to_string,
  unevaluated,
};
use num_bigint::BigInt;
use num_traits::Signed;

fn nth_prime(n: i128) -> i128 {
  if n == 0 {
    return 0; // Return 0 for invalid input
  }
  let mut count = 0;
  let mut num = 1;
  while count < n {
    num += 1;
    if is_prime_i128(num) {
      count += 1;
    }
  }
  num
}

/// True if `e` denotes a concrete number (an integer, real, big-float, or
/// rational, possibly negated) rather than a symbolic expression. Used to
/// decide whether a "not a valid index" message should fire: a concrete
/// non-integer argument is an error, but a symbolic one stays unevaluated.

/// Primality test for i128 values (Miller-Rabin via BigInt beyond 1u128 << 53).
pub fn is_prime_i128(n: i128) -> bool {
  if n < 2 {
    return false;
  }
  if n < 4 {
    return true;
  }

  if n.unsigned_abs() >= (1u128 << 53) {
    // For large integers, use Miller-Rabin instead of trial division
    let big = BigInt::from(n);
    is_prime_bigint(&big)
  } else {
    if n % 2 == 0 || n % 3 == 0 {
      return false;
    }

    let sqrt_n = (n as f64).sqrt() as i128;
    for i in (5..=sqrt_n).step_by(6) {
      // (5 + 6k) + [1,3,4,5] -> (6 + 6k) + [0,2,3,4] -> multiple of 2 or 3, already tested
      if n % i == 0 || n % (i + 2) == 0 {
        return false;
      }
    }
    true
  }
}

/// GCD[a, b, ...] - Greatest common divisor.
///
/// Accepts integers and rationals. For rationals,
/// `GCD[a/b, c/d] = GCD[a, c] / LCM[b, d]`.
/// When GCD/LCM is given an inexact (Real/BigFloat) argument it stays
/// unevaluated, but wolframscript first warns: `<F>::exact: Argument <v> in
/// <F>[...] is not an exact number.`, naming the first inexact argument (the
/// args are already Orderless-sorted by the time we see them). Purely symbolic
/// arguments (e.g. Pi) do not trigger the message.
fn emit_gcd_lcm_exact_message(name: &str, args: &[Expr]) {
  if let Some(inexact) = args
    .iter()
    .find(|a| matches!(a, Expr::Real(_) | Expr::BigFloat(_, _)))
  {
    let call = unevaluated(name, args);
    crate::emit_message(&format!(
      "{}::exact: Argument {} in {} is not an exact number.",
      name,
      expr_to_string(inexact),
      expr_to_string(&call),
    ));
  }
}

/// Round p/d (d > 0) to the nearest integer, breaking ties to even (matching
/// wolframscript's Round). Used for Gaussian-integer division.
fn round_half_even_div(p: i128, d: i128) -> i128 {
  let q = p.div_euclid(d);
  let r = p.rem_euclid(d); // 0 <= r < d
  let twice = 2 * r;
  if twice < d {
    q
  } else if twice > d {
    q + 1
  } else if q % 2 == 0 {
    q
  } else {
    q + 1
  }
}

/// Remainder of the Gaussian division a / b (a, b as (re, im) integer pairs):
/// r = a - b*Round[a/b], with the quotient rounded component-wise. `b` must be
/// nonzero.
fn gaussian_rem(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  let (ar, ai) = a;
  let (br, bi) = b;
  let denom = br * br + bi * bi;
  // a * conj(b) = (ar*br + ai*bi) + (ai*br - ar*bi) i
  let q_re = round_half_even_div(ar * br + ai * bi, denom);
  let q_im = round_half_even_div(ai * br - ar * bi, denom);
  // b*q = (br*q_re - bi*q_im) + (br*q_im + bi*q_re) i
  (ar - (br * q_re - bi * q_im), ai - (br * q_im + bi * q_re))
}

/// Normalise a nonzero Gaussian integer to its canonical associate with
/// Re > 0 and Im >= 0 (multiplying by a unit 1, i, -1, or -i), matching
/// wolframscript's GCD representative. `(0, 0)` maps to itself.
fn gaussian_normalize(mut g: (i128, i128)) -> (i128, i128) {
  if g == (0, 0) {
    return g;
  }
  for _ in 0..4 {
    let (re, im) = g;
    if re > 0 && im >= 0 {
      return g;
    }
    g = (-im, re); // multiply by i
  }
  g
}

/// GCD of two Gaussian integers via the Euclidean algorithm over Z[i].
fn gaussian_gcd(mut a: (i128, i128), mut b: (i128, i128)) -> (i128, i128) {
  while b != (0, 0) {
    let r = gaussian_rem(a, b);
    a = b;
    b = r;
  }
  gaussian_normalize(a)
}

/// Product of two Gaussian integers.
fn gaussian_mul(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  let (ar, ai) = a;
  let (br, bi) = b;
  (ar * br - ai * bi, ar * bi + ai * br)
}

/// Exact Gaussian division a / b, assuming b divides a evenly (b nonzero).
fn gaussian_exact_div(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  let (ar, ai) = a;
  let (br, bi) = b;
  let denom = br * br + bi * bi;
  // a * conj(b) / denom
  ((ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom)
}

/// LCM of two Gaussian integers: (a / gcd(a, b)) * b, normalised to the
/// canonical associate. `(0, 0)` if either operand is zero.
fn gaussian_lcm(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  if a == (0, 0) || b == (0, 0) {
    return (0, 0);
  }
  let g = gaussian_gcd(a, b);
  gaussian_normalize(gaussian_mul(gaussian_exact_div(a, g), b))
}

/// Extract a Gaussian integer (re, im) from an expression, or None if it is not
/// an exact complex number with integer real and imaginary parts.
fn expr_to_gaussian_int(expr: &Expr) -> Option<(i128, i128)> {
  let ((rn, rd), (in_, id)) =
    crate::functions::math_ast::try_extract_complex_exact(expr)?;
  if rd == 1 && id == 1 {
    Some((rn, in_))
  } else {
    None
  }
}

pub fn gcd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Integer(0));
  }

  // Gaussian-integer GCD: when at least one argument is a complex (Gaussian)
  // integer, run the Euclidean algorithm over Z[i] and return the canonical
  // associate (Re > 0, Im >= 0), e.g. GCD[7 + 3 I, 2] = 1 + I.
  if let Some(gs) = args
    .iter()
    .map(expr_to_gaussian_int)
    .collect::<Option<Vec<_>>>()
    && gs.iter().any(|(_, im)| *im != 0)
  {
    let mut g = (0i128, 0i128);
    for &z in &gs {
      g = gaussian_gcd(g, z);
    }
    let (re, im) = g;
    if im == 0 {
      return Ok(Expr::Integer(re));
    }
    let complex = Expr::FunctionCall {
      name: "Complex".to_string(),
      args: vec![Expr::Integer(re), Expr::Integer(im)].into(),
    };
    return crate::evaluator::evaluate_expr_to_expr(&complex);
  }

  // Convert each arg to (numerator, denominator); bail out for non-numeric.
  let mut fractions: Vec<(BigInt, BigInt)> = Vec::with_capacity(args.len());
  for arg in args {
    match expr_to_fraction(arg) {
      Some(f) => fractions.push(f),
      None => {
        emit_gcd_lcm_exact_message("GCD", args);
        return Ok(unevaluated("GCD", args));
      }
    }
  }

  // GCD over the numerators; LCM over the denominators.
  let mut num_gcd = fractions[0].0.clone().abs();
  let mut den_lcm = fractions[0].1.clone().abs();
  for (n, d) in &fractions[1..] {
    num_gcd = gcd_bigint(&num_gcd, n);
    den_lcm = lcm_bigint(den_lcm, d.clone());
  }

  Ok(make_rational_expr(num_gcd, den_lcm))
}

/// Extended Euclidean algorithm: returns (gcd, s, t) where a*s + b*t = gcd
fn extended_gcd_bigint(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
  use num_traits::Zero;
  if b.is_zero() {
    if a.is_zero() {
      return (BigInt::from(0), BigInt::from(0), BigInt::from(0));
    }
    let sign = if a >= &BigInt::from(0) {
      BigInt::from(1)
    } else {
      BigInt::from(-1)
    };
    return (a.abs(), sign, BigInt::from(0));
  }
  let (mut old_r, mut r) = (a.clone(), b.clone());
  let (mut old_s, mut s) = (BigInt::from(1), BigInt::from(0));
  let (mut old_t, mut t) = (BigInt::from(0), BigInt::from(1));

  while !r.is_zero() {
    let q = &old_r / &r;
    let new_r = &old_r - &q * &r;
    old_r = r;
    r = new_r;
    let new_s = &old_s - &q * &s;
    old_s = s;
    s = new_s;
    let new_t = &old_t - &q * &t;
    old_t = t;
    t = new_t;
  }

  // Ensure gcd is positive
  if old_r < BigInt::from(0) {
    old_r = -old_r;
    old_s = -old_s;
    old_t = -old_t;
  }
  (old_r, old_s, old_t)
}

/// ExtendedGCD[a, b, ...] - Extended greatest common divisor
pub fn extended_gcd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "ExtendedGCD expects at least 2 arguments".into(),
    ));
  }

  // Convert all args to BigInt
  let mut vals = Vec::new();
  for arg in args {
    let val = match arg {
      Expr::Integer(n) => BigInt::from(*n),
      Expr::BigInteger(n) => n.clone(),
      _ => {
        return Ok(unevaluated("ExtendedGCD", args));
      }
    };
    vals.push(val);
  }

  if vals.len() == 2 {
    let (g, s, t) = extended_gcd_bigint(&vals[0], &vals[1]);
    return Ok(Expr::List(
      vec![
        bigint_to_expr(g),
        Expr::List(vec![bigint_to_expr(s), bigint_to_expr(t)].into()),
      ]
      .into(),
    ));
  }

  // Multi-argument: iteratively compute
  // ExtendedGCD[a1, a2, ..., an]
  // gcd = gcd(a1, ..., an), coefficients c_i such that sum(a_i * c_i) = gcd
  // We compute iteratively: g = gcd(a1, a2), then gcd(g, a3), etc.
  // At each step, track the Bézout coefficients.
  let (mut g, s0, t0) = extended_gcd_bigint(&vals[0], &vals[1]);
  let mut coeffs = vec![s0, t0];

  for val in &vals[2..] {
    let (new_g, s, t) = extended_gcd_bigint(&g, val);
    // g_old = sum(a_i * coeffs[i]), so new relationship:
    // new_g = g_old * s + val * t = sum(a_i * coeffs[i] * s) + val * t
    for c in &mut coeffs {
      *c = &*c * &s;
    }
    coeffs.push(t);
    g = new_g;
  }

  Ok(Expr::List(
    vec![
      bigint_to_expr(g),
      Expr::List(coeffs.into_iter().map(bigint_to_expr).collect()),
    ]
    .into(),
  ))
}

/// LCM[a, b, ...] - Least common multiple.
///
/// Accepts integers and rationals. For rationals,
/// `LCM[a/b, c/d] = LCM[a, c] / GCD[b, d]`.
pub fn lcm_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // LCM requires at least one argument: wolframscript emits LCM::argm and
  // returns unevaluated (unlike GCD[], which is the identity 0).
  if args.is_empty() {
    crate::emit_message(
      "LCM::argm: LCM called with 0 arguments; 1 or more arguments are expected.",
    );
    return Ok(Expr::FunctionCall {
      name: "LCM".to_string(),
      args: vec![].into(),
    });
  }

  // Gaussian-integer LCM: when at least one argument is a complex (Gaussian)
  // integer, combine via (a/gcd)*b over Z[i] and return the canonical associate
  // (Re > 0, Im >= 0), e.g. LCM[1 + I, 1 - I] = 1 + I.
  if let Some(gs) = args
    .iter()
    .map(expr_to_gaussian_int)
    .collect::<Option<Vec<_>>>()
    && gs.iter().any(|(_, im)| *im != 0)
  {
    let mut g = gs[0];
    for &z in &gs[1..] {
      g = gaussian_lcm(g, z);
    }
    let (re, im) = gaussian_normalize(g);
    if im == 0 {
      return Ok(Expr::Integer(re));
    }
    let complex = Expr::FunctionCall {
      name: "Complex".to_string(),
      args: vec![Expr::Integer(re), Expr::Integer(im)].into(),
    };
    return crate::evaluator::evaluate_expr_to_expr(&complex);
  }

  let mut fractions: Vec<(BigInt, BigInt)> = Vec::with_capacity(args.len());
  for arg in args {
    match expr_to_fraction(arg) {
      Some(f) => fractions.push(f),
      None => {
        emit_gcd_lcm_exact_message("LCM", args);
        return Ok(unevaluated("LCM", args));
      }
    }
  }

  let mut num_lcm = fractions[0].0.clone().abs();
  let mut den_gcd = fractions[0].1.clone().abs();
  for (n, d) in &fractions[1..] {
    num_lcm = lcm_bigint(num_lcm, n.clone());
    den_gcd = gcd_bigint(&den_gcd, d);
  }

  Ok(make_rational_expr(num_lcm, den_gcd))
}

/// Convert an Expr to (numerator, denominator) if it's an integer, big
/// integer, or `Rational[p, q]`. Returns None otherwise.
fn expr_to_fraction(expr: &Expr) -> Option<(BigInt, BigInt)> {
  match expr {
    Expr::Integer(n) => Some((BigInt::from(*n), BigInt::from(1))),
    Expr::BigInteger(n) => Some((n.clone(), BigInt::from(1))),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      let num = match &args[0] {
        Expr::Integer(n) => BigInt::from(*n),
        Expr::BigInteger(n) => n.clone(),
        _ => return None,
      };
      let den = match &args[1] {
        Expr::Integer(n) => BigInt::from(*n),
        Expr::BigInteger(n) => n.clone(),
        _ => return None,
      };
      Some((num, den))
    }
    _ => None,
  }
}

/// LCM helper for BigInts. LCM(0, _) = LCM(_, 0) = 0.
fn lcm_bigint(a: BigInt, b: BigInt) -> BigInt {
  use num_traits::Zero;
  if a.is_zero() || b.is_zero() {
    return BigInt::from(0);
  }
  let g = gcd_bigint(&a, &b);
  (a.abs() / g) * b.abs()
}

/// Reduce numerator/denominator and emit either an Integer (when the
/// denominator is 1) or a `Rational[p, q]` Expr.
pub fn make_rational_expr(num: BigInt, den: BigInt) -> Expr {
  use num_traits::{One, Zero};
  if den.is_zero() {
    // Wolfram returns ComplexInfinity for 1/0 — preserve that here so
    // callers see the same surface behaviour.
    return Expr::Identifier("ComplexInfinity".to_string());
  }
  let g = gcd_bigint(&num, &den);
  let mut n = num / &g;
  let mut d = den / g;
  if d < BigInt::from(0) {
    n = -n;
    d = -d;
  }
  if d.is_one() {
    return bigint_to_expr(n);
  }
  Expr::FunctionCall {
    name: "Rational".to_string(),
    args: vec![bigint_to_expr(n), bigint_to_expr(d)].into(),
  }
}

/// Split-recursive factorial helper based on Fateman's `kg` algorithm.
/// k(n, m) computes n * (n-m) * (n-2m) * … (down to a value ≤ m).
/// When n is even and m > 1, factors of 2 are extracted and deferred
/// to a single left-shift at the end, avoiding redundant bignum work.
/// See Fateman, "Comments on Factorial Programs" (2006).
fn kg_inner(n: i128, m: i128, shift: &mut u64) -> BigInt {
  if n & 1 == 0 && m > 1 {
    // n is even, m > 1: factor out 2s — k(n,m) = 2^(n/2) * k(n/2, m/2)
    *shift += (n >> 1) as u64;
    kg_inner(n >> 1, m >> 1, shift)
  } else if n <= m {
    BigInt::from(n)
  } else {
    kg_inner(n, m << 1, shift) * kg_inner(n - m, m << 1, shift)
  }
}

/// Extract a finite f64 value from a `Real` or `BigFloat`. Returns the
/// numeric value plus the BigFloat precision marker when the input was a
/// BigFloat, so the caller can re-tag the result.
fn factorial2_extract_real(expr: &Expr) -> Option<(f64, Option<f64>)> {
  match expr {
    Expr::Real(f) => Some((*f, None)),
    Expr::BigFloat(digits, prec) => {
      digits.parse::<f64>().ok().map(|v| (v, Some(*prec)))
    }
    _ => None,
  }
}

/// Returns true if the expression contains a Real or BigFloat anywhere
/// in the tree — i.e., the value is inexact.

/// Factorial[n] or n!
/// Factorial[n] = Gamma[n+1]
/// For negative integers, returns ComplexInfinity.
/// For floats, computes Gamma[n+1] numerically.
/// For half-integer rationals, computes Gamma[n+1] symbolically.
pub fn factorial_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Factorial expects exactly 1 argument".into(),
    ));
  }
  if let Some(n) = expr_to_i128(&args[0]) {
    if n < 0 {
      // Factorial of negative integers is ComplexInfinity
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    if n <= 1 {
      return Ok(Expr::Integer(1));
    }
    let mut shift: u64 = 0;
    let result = kg_inner(n, 1, &mut shift);
    Ok(bigint_to_expr(result << shift))
  } else if let Expr::Real(f) = &args[0] {
    // Factorial has poles at the negative integers: Factorial[-n.] =
    // Gamma[1 - n] = ComplexInfinity. The numeric gamma_fn returns a large
    // finite garbage value at those poles instead of diverging, so detect
    // them explicitly. (A negative non-integer, e.g. -1.5, stays finite.)
    if *f < 0.0 && f.fract() == 0.0 {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    // An integer-valued real index gives the exact factorial rounded to a
    // machine real: Factorial[5.0] -> 120., not the float-Gamma
    // 120.00000000000021. Compute the exact integer factorial, then convert
    // to f64 so the rounding matches wolframscript's exact-then-round.
    // (170! is the largest factorial below the f64 range; beyond that
    // wolframscript switches to arbitrary precision, left to the Gamma path.)
    if f.fract() == 0.0 && *f >= 0.0 && *f <= 170.0 {
      use num_traits::ToPrimitive;
      let n = *f as i128;
      let exact = if n <= 1 {
        BigInt::from(1)
      } else {
        let mut shift: u64 = 0;
        kg_inner(n, 1, &mut shift) << shift
      };
      return Ok(Expr::Real(exact.to_f64().unwrap_or(f64::INFINITY)));
    }
    // Factorial[x] = Gamma[x+1] for real numbers
    let result = super::gamma_fn(*f + 1.0);
    if result.is_infinite() {
      Ok(Expr::Identifier("ComplexInfinity".to_string()))
    } else {
      Ok(Expr::Real(result))
    }
  } else if let Some((re, im)) = super::try_extract_complex_float(&args[0])
    && im != 0.0
    && contains_inexact_real(&args[0])
  {
    // Complex floating-point argument: Factorial[z] = Gamma[z+1].
    let (gr, gi) =
      crate::functions::math_ast::zeta_functions::gamma_complex(re + 1.0, im);
    Ok(super::build_complex_float_expr(gr, gi))
  } else {
    // For rationals and other symbolic expressions, delegate to Gamma[n+1]
    // Build the expression Gamma[n + 1] and evaluate via gamma_ast
    let n_plus_1 = match &args[0] {
      // Handle Rational[p, q] — compute (p + q) / q = Rational[p + q, q]
      Expr::FunctionCall { name, args: ra }
        if name == "Rational" && ra.len() == 2 =>
      {
        if let (Expr::Integer(p), Expr::Integer(q)) = (&ra[0], &ra[1]) {
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(p + q), Expr::Integer(*q)].into(),
          }
        } else {
          // Can't simplify, return unevaluated
          return Ok(unevaluated("Factorial", args));
        }
      }
      _ => {
        // Return unevaluated for other symbolic arguments
        return Ok(unevaluated("Factorial", args));
      }
    };
    super::gamma_ast(&[n_plus_1])
  }
}

/// Factorial2[n] - Double factorial: n!! = n * (n-2) * (n-4) * ...
pub fn factorial2_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Factorial2 expects exactly 1 argument".into(),
    ));
  }
  if let Some(n) = expr_to_i128(&args[0]) {
    if n < -1 {
      // Negative odd integers: n!! = (n+2)!! / (n+2)
      // (-1)!! = 1, (-3)!! = -1, (-5)!! = 1/3, (-7)!! = -1/15, ...
      if n % 2 != 0 {
        // (-n)!! = (-1)^((n-1)/2) / (n-2)!! for n>1 odd
        let mut denom = BigInt::from(1);
        let mut i = -n - 2;
        while i >= 2 {
          denom *= i;
          i -= 2;
        }

        let numer = BigInt::from(if (-n + 1) % 4 == 0 { -1 } else { 1 });
        return Ok(make_rational_expr(numer, denom));
      }
      return Ok(unevaluated("Factorial2", args));
    }
    if n == -1 || n == 0 {
      return Ok(Expr::Integer(1));
    }
    let mut result = BigInt::from(1);
    let mut i = n;
    while i >= 2 {
      result *= i;
      i -= 2;
    }
    Ok(bigint_to_expr(result))
  } else if let Some((x, bf_prec)) = factorial2_extract_real(&args[0]) {
    // Real argument: evaluate via the analytic continuation used by
    // Wolfram. For all real x:
    //   x!! = 2^(x/2 + (1 - Cos[π x])/4) * Gamma[x/2 + 1]
    //         / π^((1 - Cos[π x])/4)
    // This reduces to the integer recurrence when x is a non-negative
    // integer and coincides with wolframscript on non-integer reals.
    let shift = (1.0 - (std::f64::consts::PI * x).cos()) / 4.0;
    let pow2 = 2.0f64.powf(x / 2.0 + shift);
    let gamma = super::gamma_fn(x / 2.0 + 1.0);
    let pi_pow = std::f64::consts::PI.powf(shift);
    let result = pow2 * gamma / pi_pow;
    if result.is_nan() || result.is_infinite() {
      return Ok(unevaluated("Factorial2", args));
    }
    // BigFloat input → BigFloat result at the same precision marker, so
    // `N[Pi!!, 6]` lands on a precision-tagged real instead of falling back
    // to the unevaluated form.
    if let Some(prec) = bf_prec {
      let display_str = format!("{:?}", result);
      return Ok(Expr::BigFloat(display_str, prec));
    }
    Ok(Expr::Real(result))
  } else {
    Ok(unevaluated("Factorial2", args))
  }
}

/// Subfactorial[n] - Count of derangements: !n = n! * Sum[(-1)^k/k!, {k, 0, n}]
pub fn subfactorial_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Subfactorial expects exactly 1 argument".into(),
    ));
  }
  // Handle float args that are whole numbers (e.g., 6.0 → 6)
  // Track whether the input was a float so we return a float result.
  let is_float_input = matches!(&args[0], Expr::Real(_));
  let n_opt = expr_to_i128(&args[0]).or_else(|| {
    if let Expr::Real(f) = &args[0]
      && f.fract() == 0.0
      && *f >= 0.0
      && *f <= i128::MAX as f64
    {
      return Some(*f as i128);
    }
    None
  });
  if let Some(n) = n_opt {
    if n < 0 {
      return Ok(unevaluated("Subfactorial", args));
    }
    let result = if n == 0 {
      Expr::Integer(1)
    } else if n == 1 {
      Expr::Integer(0)
    } else {
      // Use recurrence: !n = (n-1) * (!(n-1) + !(n-2))
      let mut prev2 = BigInt::from(1); // !0 = 1
      let mut prev1 = BigInt::from(0); // !1 = 0
      for i in 2..=n {
        let current = BigInt::from(i - 1) * (&prev1 + &prev2);
        prev2 = prev1;
        prev1 = current;
      }
      bigint_to_expr(prev1)
    };
    if is_float_input {
      // Convert integer result to Real for float inputs (matches Wolfram).
      let f = match &result {
        Expr::Integer(i) => *i as f64,
        _ => return Ok(result),
      };
      Ok(Expr::Real(f))
    } else {
      Ok(result)
    }
  } else {
    Ok(unevaluated("Subfactorial", args))
  }
}

/// LucasL[n] - Lucas number L_n.
/// LucasL[n, x] - Lucas polynomial: L_0(x) = 2, L_1(x) = x,
/// L_{n+1}(x) = x * L_n(x) + L_{n-1}(x). Stays symbolic when n is not
/// a non-negative integer; series-expansion is wired up in
/// `series_ast` for non-integer n via the closed form
/// `((x + Sqrt[x^2 + 4])/2)^n`.
pub fn lucas_l_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "LucasL expects 1 or 2 arguments".into(),
    ));
  }

  if args.len() == 2 {
    // Two-argument form: Lucas polynomial in x.
    // Negative index uses the reflection LucasL[-n, x] = (-1)^n LucasL[n, x].
    let n_raw = match expr_to_i128(&args[0]) {
      Some(n) => n,
      None => {
        return Ok(unevaluated("LucasL", args));
      }
    };
    let negate = n_raw < 0 && n_raw % 2 != 0;
    let n = n_raw.unsigned_abs() as usize;
    let x = &args[1];
    let apply_sign = |e: Expr| -> Result<Expr, InterpreterError> {
      if negate {
        crate::evaluator::evaluate_function_call_ast(
          "Times",
          &[Expr::Integer(-1), e],
        )
      } else {
        Ok(e)
      }
    };
    // L_0(x) = 2; L_1(x) = x.
    if n == 0 {
      return Ok(Expr::Integer(2));
    }
    if n == 1 {
      return apply_sign(x.clone());
    }
    // Build via the recurrence symbolically, expanding after each step
    // so `LucasL[5, y]` returns `5*y + 5*y^3 + y^5` rather than a
    // nested-Times tree.
    let mut prev: Expr = Expr::Integer(2);
    let mut curr: Expr = x.clone();
    for _ in 2..=n {
      let next = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![x.clone(), curr.clone()].into(),
          },
          prev,
        ]
        .into(),
      };
      let expanded =
        crate::evaluator::evaluate_function_call_ast("Expand", &[next])?;
      prev = curr;
      curr = expanded;
    }
    let signed = apply_sign(curr)?;
    // The step-wise Expand leaves a rational argument's terms un-summed
    // (LucasL[4, 1/3] -> 20/9 + 19/81); evaluate so they fold to a single
    // value (199/81). A symbolic polynomial is already canonical, so this is
    // idempotent for it.
    return crate::evaluator::evaluate_expr_to_expr(&signed);
  }

  // Real argument: analytic continuation
  // LucasL[x] = phi^x + Cos[pi x] phi^-x (phi = golden ratio).
  if let Expr::Real(x) = &args[0] {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let val = phi.powf(*x) + (std::f64::consts::PI * *x).cos() * phi.powf(-*x);
    return Ok(Expr::Real(val));
  }

  // Negative index uses the reflection LucasL[-n] = (-1)^n LucasL[n].
  let n_raw = match expr_to_i128(&args[0]) {
    Some(n) => n,
    None => {
      return Ok(unevaluated("LucasL", args));
    }
  };
  let negate = n_raw < 0 && n_raw % 2 != 0;
  let n = n_raw.unsigned_abs() as usize;
  // L(0) = 2, L(1) = 1, L(n) = L(n-1) + L(n-2)
  if n == 0 {
    return Ok(Expr::Integer(2));
  }
  let result = if n == 1 {
    BigInt::from(1)
  } else {
    let mut a = BigInt::from(2);
    let mut b = BigInt::from(1);
    for _ in 2..=n {
      let c = &a + &b;
      a = b;
      b = c;
    }
    b
  };
  Ok(bigint_to_expr(if negate { -result } else { result }))
}

/// ChineseRemainder[{r1,r2,...}, {m1,m2,...}] - Chinese Remainder Theorem
pub fn chinese_remainder_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 && args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "ChineseRemainder expects 2 or 3 arguments".into(),
    ));
  }
  let uneval = || Ok(unevaluated("ChineseRemainder", args));
  // The optional third argument shifts the result into [d, d + lcm).
  let offset = if args.len() == 3 {
    match expr_to_i128(&args[2]) {
      Some(d) => Some(d),
      None => return uneval(),
    }
  } else {
    None
  };
  let remainders = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(unevaluated("ChineseRemainder", args));
    }
  };
  let moduli = match &args[1] {
    Expr::List(items) => items,
    _ => {
      return Ok(unevaluated("ChineseRemainder", args));
    }
  };
  if remainders.len() != moduli.len() {
    return Err(InterpreterError::EvaluationError(
      "ChineseRemainder: lists must have the same length".into(),
    ));
  }

  let mut r_vals = Vec::new();
  let mut m_vals = Vec::new();
  for (r, m) in remainders.iter().zip(moduli.iter()) {
    match (r, m) {
      (r, m) if expr_to_i128(r).is_some() && expr_to_i128(m).is_some() => {
        let mv = expr_to_i128(m).unwrap();
        if mv > 0 {
          r_vals.push(expr_to_i128(r).unwrap());
          m_vals.push(mv);
        } else {
          return Ok(unevaluated("ChineseRemainder", args));
        }
      }
      _ => {
        return Ok(unevaluated("ChineseRemainder", args));
      }
    }
  }

  // Extended GCD helper
  fn extended_gcd(a: i128, b: i128) -> (i128, i128, i128) {
    if b == 0 {
      (a, 1, 0)
    } else {
      let (g, x1, y1) = extended_gcd(b, a % b);
      (g, y1, x1 - (a / b) * y1)
    }
  }

  // Solve using CRT iteratively
  let mut result = r_vals[0].rem_euclid(m_vals[0]);
  let mut modulus = m_vals[0];

  for i in 1..r_vals.len() {
    let ri = r_vals[i].rem_euclid(m_vals[i]);
    let mi = m_vals[i];
    let (g, p, _) = extended_gcd(modulus, mi);
    if (ri - result) % g != 0 {
      // Incompatible congruences: no solution. wolframscript leaves the call
      // unevaluated rather than erroring.
      return uneval();
    }
    let lcm = modulus / g * mi;
    result =
      (result + modulus * ((ri - result) / g % (mi / g)) * p).rem_euclid(lcm);
    modulus = lcm;
  }

  // The 2-argument form returns the smallest non-negative solution (in
  // [0, modulus)); the 3-argument form returns the smallest solution >= d
  // (in [d, d + modulus)).
  match offset {
    None => Ok(Expr::Integer(result)),
    Some(d) => Ok(Expr::Integer(d + (result - d).rem_euclid(modulus))),
  }
}

/// DivisorSum[n, form] — applies form to each divisor and sums
/// DivisorSum[n, form, cond] — also filters by cond[div]
pub fn divisor_sum_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "DivisorSum expects 2 or 3 arguments".into(),
    ));
  }
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n > 0 => n,
    _ => {
      return Ok(unevaluated("DivisorSum", args));
    }
  };
  let func = &args[1];
  let cond = if args.len() == 3 {
    Some(&args[2])
  } else {
    None
  };

  // Get divisors via fast factorization (the O(√n) trial-division path hangs
  // on the ~10^20 terms of RosettaCode's aliquot-sequence-classifications,
  // even though those terms have very few divisors).
  let n_u = n as u128;
  let divs: Vec<i128> = match divisors_u128(n_u) {
    Some(d) => d.into_iter().map(|x| x as i128).collect(),
    None => {
      // Fallback: O(√n) trial division.
      let sqrt_n = (n_u as f64).sqrt() as u128;
      let mut small_divs: Vec<u128> = Vec::new();
      let mut large_divs: Vec<u128> = Vec::new();
      for i in 1..=sqrt_n {
        if n_u.is_multiple_of(i) {
          small_divs.push(i);
          if i != n_u / i {
            large_divs.push(n_u / i);
          }
        }
      }
      large_divs.reverse();
      let mut divs: Vec<i128> =
        Vec::with_capacity(small_divs.len() + large_divs.len());
      divs.extend(small_divs.iter().map(|&x| x as i128));
      divs.extend(large_divs.iter().map(|&x| x as i128));
      divs
    }
  };

  // Apply function to each divisor and sum, filtering by cond if present
  let mut sum = Expr::Integer(0);
  for d in divs {
    if let Some(c) = cond {
      let keep = crate::evaluator::apply_function_to_arg(c, &Expr::Integer(d))?;
      let keep = crate::evaluator::evaluate_expr_to_expr(&keep)?;
      if !matches!(&keep, Expr::Identifier(s) if s == "True") {
        continue;
      }
    }
    let val = crate::evaluator::apply_function_to_arg(func, &Expr::Integer(d))?;
    sum = super::plus_ast(&[sum, val])?;
  }
  Ok(sum)
}

// ─── Combinatorics Functions ─────────────────────────────────────

/// BernoulliB[n] - nth Bernoulli number
/// BernoulliB[n, z] - nth Bernoulli polynomial
pub fn bernoulli_b_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "BernoulliB expects 1 or 2 arguments".into(),
    ));
  }
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n as usize,
    _ => {
      let call = unevaluated("BernoulliB", args);
      // A concrete first argument that isn't a non-negative integer (a
      // negative integer, a real, or a rational) emits intnm; a symbolic
      // argument stays unevaluated silently. The two-argument polynomial
      // form never emits this message.
      if args.len() == 1 && is_concrete_number(&args[0]) {
        crate::emit_message(&format!(
          "BernoulliB::intnm: Non-negative machine-sized integer expected at position 1 in {}.",
          crate::syntax::expr_to_message_form(&call)
        ));
      }
      return Ok(call);
    }
  };

  if args.len() == 2 {
    return bernoulli_polynomial(n, &args[1]);
  }

  // Use the formula: B(n) computed via the explicit sum formula
  // Store as rational numbers (numer, denom)
  // B(0) = 1, B(1) = -1/2, B(odd>1) = 0
  if n == 0 {
    return Ok(Expr::Integer(1));
  }
  if n == 1 {
    return Ok(make_rational(-1, 2));
  }
  if n % 2 != 0 {
    return Ok(Expr::Integer(0));
  }

  // Compute using the recurrence: sum_{k=0}^{n-1} C(n,k) * B(k) / (n - k + 1) = 0 ... wait
  // Better: B(n) = -1/(n+1) * sum_{k=0}^{n-1} C(n+1, k) * B(k)
  // We'll compute all Bernoulli numbers up to n

  // Represent each Bernoulli number as a reduced (numerator, denominator)
  // pair in BigInt, so large numerators (e.g. BernoulliB[60]) don't overflow
  // i128.
  fn reduce(num: BigInt, den: BigInt) -> (BigInt, BigInt) {
    let mut g = gcd_bigint(&num, &den);
    if g == BigInt::from(0) {
      g = BigInt::from(1);
    }
    let (mut num, mut den) = (num / &g, den / &g);
    if den < BigInt::from(0) {
      num = -num;
      den = -den;
    }
    (num, den)
  }
  fn rat_add(a: &(BigInt, BigInt), b: &(BigInt, BigInt)) -> (BigInt, BigInt) {
    reduce(&a.0 * &b.1 + &b.0 * &a.1, &a.1 * &b.1)
  }
  fn rat_mul(a: &(BigInt, BigInt), b: &(BigInt, BigInt)) -> (BigInt, BigInt) {
    reduce(&a.0 * &b.0, &a.1 * &b.1)
  }

  let mut b: Vec<(BigInt, BigInt)> = Vec::with_capacity(n + 1);
  b.push((BigInt::from(1), BigInt::from(1))); // B(0) = 1
  if n >= 1 {
    b.push((BigInt::from(-1), BigInt::from(2))); // B(1) = -1/2
  }

  for m in 2..=n {
    if m % 2 != 0 {
      b.push((BigInt::from(0), BigInt::from(1)));
      continue;
    }
    // B(m) = -1/(m+1) * sum_{k=0}^{m-1} C(m+1, k) * B(k)
    let mut sum = (BigInt::from(0), BigInt::from(1));
    for k in 0..m {
      let binom = binomial_coeff_big((m + 1) as i128, k as i128);
      sum = rat_add(&sum, &rat_mul(&(binom, BigInt::from(1)), &b[k]));
    }
    let neg_inv = (BigInt::from(-1), BigInt::from(m as i128 + 1));
    b.push(rat_mul(&neg_inv, &sum));
  }

  let (num, den) = b[n].clone();
  Ok(make_rational_expr(num, den))
}

/// Compute the nth Bernoulli polynomial B_n(z) = sum_{k=0}^{n} C(n,k) * B_k * z^(n-k)
fn bernoulli_polynomial(n: usize, z: &Expr) -> Result<Expr, InterpreterError> {
  fn rat_add(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
    let num = a.0 * b.1 + b.0 * a.1;
    let den = a.1 * b.1;
    let g = gcd(num, den);
    if den < 0 {
      (-num / g, -den / g)
    } else {
      (num / g, den / g)
    }
  }

  fn rat_mul(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
    let num = a.0 * b.0;
    let den = a.1 * b.1;
    let g = gcd(num, den);
    if den < 0 {
      (-num / g, -den / g)
    } else {
      (num / g, den / g)
    }
  }

  // First compute all Bernoulli numbers B_0 through B_n as rationals
  let mut b_nums: Vec<(i128, i128)> = Vec::with_capacity(n + 1);
  b_nums.push((1, 1)); // B(0) = 1
  if n >= 1 {
    b_nums.push((-1, 2)); // B(1) = -1/2
  }
  for m in 2..=n {
    if m % 2 != 0 && m > 1 {
      b_nums.push((0, 1));
      continue;
    }
    let mut sum: (i128, i128) = (0, 1);
    let mut binom: i128 = 1;
    for k in 0..m {
      sum = rat_add(sum, rat_mul((binom, 1), b_nums[k]));
      binom = binom * (m as i128 + 1 - k as i128) / (k as i128 + 1);
    }
    b_nums.push(rat_mul((-1, m as i128 + 1), sum));
  }

  // Compute polynomial coefficients: coeff of z^(n-k) = C(n,k) * B_k
  // We'll store as coeffs[j] = coefficient of z^j
  let mut coeffs: Vec<(i128, i128)> = vec![(0, 1); n + 1];
  let mut binom: i128 = 1; // C(n, k)
  for k in 0..=n {
    let j = n - k; // power of z
    coeffs[j] = rat_mul((binom, 1), b_nums[k]);
    if k < n {
      binom = binom * (n as i128 - k as i128) / (k as i128 + 1);
    }
  }

  // If z is numeric, evaluate directly
  if let Some(z_val) = expr_to_rational(z) {
    let (z_num, z_den) = z_val;
    let mut result = (0i128, 1i128);
    for k in (0..=n).rev() {
      let rn = result.0 * z_num;
      let rd = result.1 * z_den;
      let g = gcd(rn, rd);
      let (rn, rd) = if rd < 0 {
        (-rn / g, -rd / g)
      } else {
        (rn / g, rd / g)
      };
      result = rat_add((rn, rd), coeffs[k]);
    }
    return Ok(make_rational(result.0, result.1));
  }

  // Build symbolic polynomial
  let mut terms = Vec::new();
  for k in 0..=n {
    let (cn, cd) = coeffs[k];
    if cn == 0 {
      continue;
    }
    let coeff_expr = make_rational(cn, cd);
    let term = if k == 0 {
      coeff_expr
    } else {
      let power = if k == 1 {
        z.clone()
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(z.clone()),
          right: Box::new(Expr::Integer(k as i128)),
        }
      };
      if cn == cd {
        power
      } else if cn == -cd {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(-1)),
          right: Box::new(power),
        }
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(coeff_expr),
          right: Box::new(power),
        }
      }
    };
    terms.push(term);
  }

  if terms.is_empty() {
    return Ok(Expr::Integer(0));
  }

  let mut result = terms[0].clone();
  for term in &terms[1..] {
    result = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(result),
      right: Box::new(term.clone()),
    };
  }
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// EulerE[n] - nth Euler number
/// EulerE[n, z] - nth Euler polynomial
pub fn euler_e_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "EulerE expects 1 or 2 arguments".into(),
    ));
  }

  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n as usize,
    _ => {
      let call = unevaluated("EulerE", args);
      // As with BernoulliB: a concrete non-(non-negative-integer) first
      // argument emits intnm; symbolic stays silent; the polynomial form
      // never emits it.
      if args.len() == 1 && is_concrete_number(&args[0]) {
        crate::emit_message(&format!(
          "EulerE::intnm: Non-negative machine-sized integer expected at position 1 in {}.",
          crate::syntax::expr_to_message_form(&call)
        ));
      }
      return Ok(call);
    }
  };

  if args.len() == 1 {
    // Euler number E(n)
    // E(0) = 1, E(odd) = 0
    // E(2m) = -sum_{k=0}^{m-1} C(2m, 2k) * E(2k)
    if n == 0 {
      return Ok(Expr::Integer(1));
    }
    if n % 2 != 0 {
      return Ok(Expr::Integer(0));
    }

    let m = n / 2;
    let mut e_even: Vec<BigInt> = Vec::with_capacity(m + 1);
    e_even.push(BigInt::from(1)); // E(0) = 1

    for i in 1..=m {
      let nn = 2 * i;
      let mut sum = BigInt::from(0);
      let mut binom = BigInt::from(1); // C(nn, 0)
      for k in 0..i {
        sum += &binom * &e_even[k];
        // C(nn, 2k+2) = C(nn, 2k) * (nn-2k)*(nn-2k-1) / ((2k+1)*(2k+2))
        binom = binom * BigInt::from(nn - 2 * k) * BigInt::from(nn - 2 * k - 1)
          / BigInt::from((2 * k + 1) * (2 * k + 2));
      }
      e_even.push(-sum);
    }

    let result = e_even[m].clone();
    // Convert BigInt to Expr
    Ok(bigint_to_expr(result))
  } else {
    // Euler polynomial E_n(z)
    // Build using integration relation:
    // E_0(x) = 1
    // E_n(x) = integral(n * E_{n-1}(x)) + C
    // where C is chosen so E_n(0) + E_n(1) = 0 for n >= 1
    //
    // Represent polynomial as vector of rational coefficients (num, den)
    // coeffs[k] = coefficient of x^k
    euler_polynomial(n, &args[1])
  }
}

/// Compute the nth Euler polynomial and either evaluate at a point or return symbolic expression
fn euler_polynomial(n: usize, z: &Expr) -> Result<Expr, InterpreterError> {
  fn rat_add(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
    let num = a.0 * b.1 + b.0 * a.1;
    let den = a.1 * b.1;
    let g = gcd(num, den);
    if den < 0 {
      (-num / g, -den / g)
    } else {
      (num / g, den / g)
    }
  }

  // Build polynomial coefficients using recurrence
  // coeffs[k] = (numerator, denominator) for coefficient of x^k
  let mut coeffs: Vec<(i128, i128)> = vec![(1, 1)]; // E_0(x) = 1

  for degree in 1..=n {
    // E_degree(x) = integral(degree * E_{degree-1}(x)) + C
    // integral of degree * sum(a_k * x^k) = sum(degree * a_k / (k+1) * x^(k+1))
    let prev_len = coeffs.len(); // should be `degree`
    let mut new_coeffs: Vec<(i128, i128)> = Vec::with_capacity(degree + 1);
    new_coeffs.push((0, 1)); // placeholder for constant term

    for k in 0..prev_len {
      // coefficient of x^(k+1) = degree * coeffs[k] / (k+1)
      let (cn, cd) = coeffs[k];
      let num = degree as i128 * cn;
      let den = cd * (k as i128 + 1);
      let g = gcd(num, den);
      let (num, den) = if den < 0 {
        (-num / g, -den / g)
      } else {
        (num / g, den / g)
      };
      new_coeffs.push((num, den));
    }

    // Determine C such that E_n(0) + E_n(1) = 0
    // E_n(0) = C
    // E_n(1) = C + sum_{k=1}^{degree} new_coeffs[k]
    // So 2C + sum_{k=1}^{degree} new_coeffs[k] = 0
    // C = -1/2 * sum_{k=1}^{degree} new_coeffs[k]
    let mut sum_at_1 = (0i128, 1i128);
    for k in 1..=degree {
      sum_at_1 = rat_add(sum_at_1, new_coeffs[k]);
    }
    // C = -sum_at_1 / 2
    let c_num = -sum_at_1.0;
    let c_den = sum_at_1.1 * 2;
    let g = gcd(c_num, c_den);
    let (c_num, c_den) = if c_den < 0 {
      (-c_num / g, -c_den / g)
    } else {
      (c_num / g, c_den / g)
    };
    new_coeffs[0] = (c_num, c_den);

    coeffs = new_coeffs;
  }

  // Now coeffs has the polynomial coefficients for E_n(z)
  // Check if z is numeric - if so, evaluate directly
  if let Some(z_val) = expr_to_rational(z) {
    // Evaluate polynomial at z_val using Horner's method
    let (z_num, z_den) = z_val;
    let mut result = (0i128, 1i128);
    for k in (0..=n).rev() {
      // result = result * z + coeffs[k]
      let rn = result.0 * z_num;
      let rd = result.1 * z_den;
      let g = gcd(rn, rd);
      let (rn, rd) = if rd < 0 {
        (-rn / g, -rd / g)
      } else {
        (rn / g, rd / g)
      };
      result = rat_add((rn, rd), coeffs[k]);
    }
    return Ok(make_rational(result.0, result.1));
  }

  // Build symbolic polynomial expression
  let mut terms = Vec::new();
  for k in 0..=n {
    let (cn, cd) = coeffs[k];
    if cn == 0 {
      continue;
    }
    let coeff_expr = make_rational(cn, cd);
    let term = if k == 0 {
      coeff_expr
    } else {
      let power = if k == 1 {
        z.clone()
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(z.clone()),
          right: Box::new(Expr::Integer(k as i128)),
        }
      };
      if cn == cd {
        // coefficient is 1
        power
      } else if cn == -cd {
        // coefficient is -1
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(-1)),
          right: Box::new(power),
        }
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(coeff_expr),
          right: Box::new(power),
        }
      }
    };
    terms.push(term);
  }

  if terms.is_empty() {
    return Ok(Expr::Integer(0));
  }

  let mut result = terms[0].clone();
  for term in &terms[1..] {
    result = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(result),
      right: Box::new(term.clone()),
    };
  }
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// BellB[n] - nth Bell number
/// BellB[n, x] - nth Bell polynomial
pub fn bell_b_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "BellB expects 1 or 2 arguments".into(),
    ));
  }

  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n as usize,
    _ => {
      let call = unevaluated("BellB", args);
      // As with BernoulliB/EulerE: a concrete first argument that isn't a
      // non-negative integer emits intnm; symbolic stays silent; the Bell
      // polynomial form never emits it.
      if args.len() == 1 && is_concrete_number(&args[0]) {
        crate::emit_message(&format!(
          "BellB::intnm: Non-negative machine-sized integer expected at position 1 in {}.",
          crate::syntax::expr_to_message_form(&call)
        ));
      }
      return Ok(call);
    }
  };

  if args.len() == 1 {
    // Bell number B_n via the Bell triangle
    if n == 0 {
      return Ok(Expr::Integer(1));
    }
    // Compute using the Bell triangle (first column of each row), in BigInt
    // so large Bell numbers (e.g. BellB[50], 48 digits) don't overflow i128.
    let zero = BigInt::from(0);
    let one = BigInt::from(1);
    let mut row = vec![zero.clone(); n + 1];
    row[0] = one.clone();
    let mut next = vec![zero.clone(); n + 1];
    for i in 1..=n {
      next[0] = &row[i - 1] - &row[0];
      for j in 1..=i {
        next[j] = &next[j - 1] + &row[j - 1];
      }
      (row, next) = (next, row);
    }

    Ok(crate::functions::math_ast::bigint_to_expr(row[n].clone()))
  } else {
    // Bell polynomial B_n(x) = sum_{k=0}^{n} S(n,k) * x^k
    // where S(n,k) is the Stirling number of the second kind
    if n == 0 {
      return Ok(Expr::Integer(1));
    }
    // Compute Stirling numbers of the second kind for all k, in BigInt so
    // large coefficients (e.g. BellB[50, x]) don't overflow i128.
    let zero = BigInt::from(0);
    let one = BigInt::from(1);
    let mut row = vec![zero.clone(); n + 1];
    row[0] = one.clone();
    let mut next = vec![zero.clone(); n + 1];
    for i in 1..=n {
      next[0] = zero.clone();
      for j in 1..=i {
        next[j] = &row[j - 1] + BigInt::from(j as i128) * &row[j];
      }
      (row, next) = (next, row);
    }
    // Build polynomial: sum_{k=0}^{n} S(n,k) * x^k
    let x = &args[1];
    let mut terms = Vec::new();
    for k in 0..=n {
      let s = row[k].clone();
      if s == zero {
        continue;
      }
      let is_one = s == one;
      let coeff = crate::functions::math_ast::bigint_to_expr(s);
      let term = if k == 0 {
        coeff
      } else if k == 1 {
        if is_one {
          x.clone()
        } else {
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(coeff),
            right: Box::new(x.clone()),
          }
        }
      } else {
        let power = Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(x.clone()),
          right: Box::new(Expr::Integer(k as i128)),
        };
        if is_one {
          power
        } else {
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(coeff),
            right: Box::new(power),
          }
        }
      };
      terms.push(term);
    }
    if terms.is_empty() {
      return Ok(Expr::Integer(0));
    }
    // Build sum of terms
    let mut result = terms[0].clone();
    for term in &terms[1..] {
      result = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(result),
        right: Box::new(term.clone()),
      };
    }
    crate::evaluator::evaluate_expr_to_expr(&result)
  }
}

/// PauliMatrix[k] - kth Pauli matrix (k=0,1,2,3)
pub fn pauli_matrix_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PauliMatrix expects exactly 1 argument".into(),
    ));
  }
  let k = match expr_to_i128(&args[0]) {
    Some(k) => k,
    _ => {
      return Ok(unevaluated("PauliMatrix", args));
    }
  };
  let i_expr = Expr::Identifier("I".to_string());
  let neg_i = Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(Expr::Integer(-1)),
    right: Box::new(i_expr.clone()),
  };
  match k {
    0 => Ok(Expr::List(
      vec![
        Expr::List(vec![Expr::Integer(1), Expr::Integer(0)].into()),
        Expr::List(vec![Expr::Integer(0), Expr::Integer(1)].into()),
      ]
      .into(),
    )),
    1 => Ok(Expr::List(
      vec![
        Expr::List(vec![Expr::Integer(0), Expr::Integer(1)].into()),
        Expr::List(vec![Expr::Integer(1), Expr::Integer(0)].into()),
      ]
      .into(),
    )),
    2 => {
      let neg_i_eval = crate::evaluator::evaluate_expr_to_expr(&neg_i)?;
      Ok(Expr::List(
        vec![
          Expr::List(vec![Expr::Integer(0), neg_i_eval].into()),
          Expr::List(vec![i_expr, Expr::Integer(0)].into()),
        ]
        .into(),
      ))
    }
    3 => Ok(Expr::List(
      vec![
        Expr::List(vec![Expr::Integer(1), Expr::Integer(0)].into()),
        Expr::List(vec![Expr::Integer(0), Expr::Integer(-1)].into()),
      ]
      .into(),
    )),
    _ => Ok(unevaluated("PauliMatrix", args)),
  }
}

/// CatalanNumber[n] - nth Catalan number = C(2n,n)/(n+1)
pub fn catalan_number_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "CatalanNumber expects exactly 1 argument".into(),
    ));
  }
  // A real argument evaluates numerically (wolframscript: CatalanNumber[3.0]
  // -> 5., CatalanNumber[2.5] -> 3.104...). An integer-valued real yields the
  // exact integer rounded to a machine real; a non-integer real uses the
  // analytic continuation CatalanNumber[z] = 4^z Gamma[z+1/2] /
  // (Sqrt[Pi] Gamma[z+2]).
  if let Expr::Real(f) = &args[0] {
    use num_traits::ToPrimitive;
    if f.fract() == 0.0 {
      let m = *f as i128;
      let exact = if m >= 0 {
        let two_m = BigInt::from(2 * m);
        let mut c = BigInt::from(1);
        for i in 0..m {
          c = c * (&two_m - BigInt::from(i)) / BigInt::from(i + 1);
        }
        c / BigInt::from(m + 1)
      } else if m == -1 {
        BigInt::from(-1)
      } else {
        BigInt::from(0)
      };
      return Ok(Expr::Real(exact.to_f64().unwrap_or(f64::INFINITY)));
    }
    let z = *f;
    let val = 4.0_f64.powf(z) * super::gamma_fn(z + 0.5)
      / (std::f64::consts::PI.sqrt() * super::gamma_fn(z + 2.0));
    return Ok(Expr::Real(val));
  }

  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n,
    // Analytic continuation collapses at negative integers: the only
    // non-zero value is CatalanNumber[-1] = -1; CatalanNumber[-n] = 0 for
    // n >= 2 (the Gamma in the numerator stays finite while the
    // denominator Gamma[2 + n] has a pole).
    Some(-1) => return Ok(Expr::Integer(-1)),
    Some(_) => return Ok(Expr::Integer(0)),
    None => {
      return Ok(unevaluated("CatalanNumber", args));
    }
  };

  // CatalanNumber[n] = Binomial[2n, n] / (n + 1), in BigInt so large results
  // (e.g. CatalanNumber[100], 57 digits) don't overflow i128.
  let result = binomial_coeff_big(2 * n, n) / BigInt::from(n + 1);
  Ok(crate::functions::math_ast::bigint_to_expr(result))
}

/// StirlingS1[n, k] - Stirling number of the first kind (signed)
/// Unevaluated Stirling call; a concrete numeric argument that is not a
/// non-negative machine integer emits `::intnm` with the offending
/// position, matching wolframscript (symbolic arguments stay silent).
fn stirling_intnm_unevaluated(fname: &str, args: &[Expr], pos: usize) -> Expr {
  let call = unevaluated(fname, args);
  if is_concrete_number(&args[pos - 1]) {
    crate::emit_message(&format!(
      "{fname}::intnm: Non-negative machine-sized integer expected at position {pos} in {}.",
      crate::syntax::expr_to_message_form(&call)
    ));
  }
  call
}

pub fn stirling_s1_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StirlingS1 expects exactly 2 arguments".into(),
    ));
  }
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n as usize,
    _ => return Ok(stirling_intnm_unevaluated("StirlingS1", args, 1)),
  };
  let k = match expr_to_i128(&args[1]) {
    Some(k) if k >= 0 => k as usize,
    _ => return Ok(stirling_intnm_unevaluated("StirlingS1", args, 2)),
  };

  if k > n {
    return Ok(Expr::Integer(0));
  }
  if n == 0 && k == 0 {
    return Ok(Expr::Integer(1));
  }
  if k == 0 {
    return Ok(Expr::Integer(0));
  }

  // s(n,k) = s(n-1,k-1) - (n-1)*s(n-1,k) (signed Stirling S1)
  // Use DP table with BigInt to avoid overflow
  let zero = BigInt::from(0);
  let one = BigInt::from(1);
  let mut row = vec![zero.clone(); k + 1];
  row[0] = one.clone();
  let mut next = vec![zero.clone(); k + 1];
  for i in 1..=n {
    next[0] = zero.clone();
    for j in 1..=k.min(i) {
      next[j] = &row[j - 1] - BigInt::from(i - 1) * &row[j];
    }
    (row, next) = (next, row);
  }
  Ok(bigint_to_expr(row[k].clone()))
}

/// StirlingS2[n, k] - Stirling number of the second kind
pub fn stirling_s2_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StirlingS2 expects exactly 2 arguments".into(),
    ));
  }
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n as usize,
    _ => return Ok(stirling_intnm_unevaluated("StirlingS2", args, 1)),
  };
  let k = match expr_to_i128(&args[1]) {
    Some(k) if k >= 0 => k as usize,
    _ => return Ok(stirling_intnm_unevaluated("StirlingS2", args, 2)),
  };

  if k > n {
    return Ok(Expr::Integer(0));
  }
  if n == 0 && k == 0 {
    return Ok(Expr::Integer(1));
  }
  if k == 0 {
    return Ok(Expr::Integer(0));
  }

  // S(n,k) = k*S(n-1,k) + S(n-1,k-1)
  let zero = BigInt::from(0);
  let one = BigInt::from(1);
  let mut row = vec![zero.clone(); k + 1];
  row[0] = one.clone();
  let mut next = vec![zero.clone(); k + 1];
  for i in 1..=n {
    next[0] = zero.clone();
    for j in 1..=k.min(i) {
      next[j] = BigInt::from(j) * &row[j] + &row[j - 1];
    }
    (row, next) = (next, row);
  }
  Ok(bigint_to_expr(row[k].clone()))
}

/// Digamma function approximation using the asymptotic series.
/// ψ(x) for x > 0 using recurrence and Stirling-like expansion.
pub fn digamma(mut x: f64) -> f64 {
  let mut result = 0.0;
  // Use recurrence ψ(x+1) = ψ(x) + 1/x to shift x to large values
  while x < 20.0 {
    result -= 1.0 / x;
    x += 1.0;
  }
  // Asymptotic expansion: ψ(x) ~ ln(x) - 1/(2x) - Σ B_{2k}/(2k·x^{2k})
  result += x.ln() - 0.5 / x;
  let x2 = x * x;
  let mut xpow = x2;
  // B_2/(2·x^2), B_4/(4·x^4), B_6/(6·x^6), ...
  let coeffs = [
    1.0 / 12.0,       // B_2/2 = 1/12
    -1.0 / 120.0,     // B_4/4 = -1/30 / 4
    1.0 / 252.0,      // B_6/6
    -1.0 / 240.0,     // B_8/8
    1.0 / 132.0,      // B_10/10
    -691.0 / 32760.0, // B_12/12
    1.0 / 12.0,       // B_14/14
  ];
  for i in 0..coeffs.len() {
    result -= coeffs[i] / xpow;
    xpow *= x2;
  }
  result
}

/// HarmonicNumber[n] - Returns the nth harmonic number H_n = 1 + 1/2 + ... + 1/n.
/// HarmonicNumber[n, r] - Returns the generalized harmonic number H_{n,r} = Sum[1/k^r, {k,1,n}].
/// True when an expression carries an inexact (machine-precision) real, so
/// HarmonicNumber should numericize. Exact arguments — integers, rationals and
/// exact irrationals like Sqrt[2] or Pi — stay symbolic, matching wolframscript
/// (HarmonicNumber[1/2] is kept unevaluated, HarmonicNumber[0.5] is not).
fn harmonic_arg_is_inexact(e: &Expr) -> bool {
  match e {
    Expr::Real(_) => true,
    Expr::FunctionCall { args, .. } => args.iter().any(harmonic_arg_is_inexact),
    Expr::BinaryOp { left, right, .. } => {
      harmonic_arg_is_inexact(left) || harmonic_arg_is_inexact(right)
    }
    Expr::UnaryOp { operand, .. } => harmonic_arg_is_inexact(operand),
    _ => false,
  }
}

pub fn harmonic_number_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "HarmonicNumber expects 1 or 2 arguments".into(),
    ));
  }

  // Infinity argument: the limiting value of the (generalized) harmonic
  // series. H[Infinity] diverges -> Infinity; H[Infinity, s] -> Zeta[s] for
  // s > 1, Infinity for 0 <= s <= 1, and Indeterminate for s < 0. Handling
  // this directly also stops Limit[HarmonicNumber[n, s], n -> Infinity] from
  // falling into the numeric fallback, which would try to sum ~10^7 terms.
  let is_infinity = matches!(&args[0],
    Expr::Identifier(n) | Expr::Constant(n) if n == "Infinity");
  if is_infinity {
    if args.len() == 1 {
      return Ok(Expr::Identifier("Infinity".to_string()));
    }
    if let Some(s) = expr_to_num(&args[1]) {
      if s > 1.0 {
        let zeta = Expr::FunctionCall {
          name: "Zeta".to_string(),
          args: vec![args[1].clone()].into(),
        };
        return crate::evaluator::evaluate_expr_to_expr(&zeta);
      } else if s >= 0.0 {
        return Ok(Expr::Identifier("Infinity".to_string()));
      } else {
        return Ok(Expr::Identifier("Indeterminate".to_string()));
      }
    }
    // Symbolic order: stay unevaluated.
    return Ok(unevaluated("HarmonicNumber", args));
  }

  // Handle inexact real argument: H(x) = digamma(x+1) + EulerGamma. Only a
  // machine-precision real numericizes; exact non-integer arguments (1/2,
  // Sqrt[2], Pi, …) stay symbolic like wolframscript.
  if args.len() == 1
    && harmonic_arg_is_inexact(&args[0])
    && let Some(x) = expr_to_num(&args[0])
  {
    // Real input - use digamma approximation
    // Euler-Mascheroni constant
    const EULER_GAMMA: f64 = 0.5772156649015329;
    let result = digamma(x + 1.0) + EULER_GAMMA;
    return Ok(Expr::Real(result));
  }

  // HarmonicNumber[n, -j] with a negative integer order is the power sum
  // Sum[k^j, {k, 1, n}] (Faulhaber), a polynomial in n. wolframscript renders
  // it expanded. Only handle a symbolic first argument here; concrete
  // non-negative integers are summed exactly below.
  if args.len() == 2
    && expr_to_num(&args[0]).is_none()
    && let Some(r) = expr_to_i128(&args[1])
    && r < 0
  {
    let j = -r;
    let k = Expr::Identifier("k".to_string());
    let summand = if j == 1 {
      k.clone()
    } else {
      // Build the BinaryOp Power form (k^j), which Sum's closed-form path
      // recognises; FunctionCall Power[k, j] would be left unevaluated.
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(k.clone()),
        right: Box::new(Expr::Integer(j)),
      }
    };
    let sum = Expr::FunctionCall {
      name: "Sum".to_string(),
      args: vec![
        summand,
        Expr::List(vec![k, Expr::Integer(1), args[0].clone()].into()),
      ]
      .into(),
    };
    let expanded = Expr::FunctionCall {
      name: "Expand".to_string(),
      args: vec![sum].into(),
    };
    return crate::evaluator::evaluate_expr_to_expr(&expanded);
  }

  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n,
    Some(_) => {
      // HarmonicNumber[n, r] = Zeta[r] - HurwitzZeta[r, n+1]. At a negative
      // integer n the Hurwitz zeta has a pole for a positive order r, so the
      // harmonic number diverges to ComplexInfinity. (For r <= 0 it is a
      // finite Faulhaber polynomial, left to the existing handling; a symbolic
      // order stays unevaluated.)
      let order_is_positive = match args.get(1) {
        None => true, // the 1-argument form has order 1
        Some(r_expr) => matches!(expr_to_i128(r_expr), Some(r) if r >= 1),
      };
      if order_is_positive {
        return Ok(Expr::Identifier("ComplexInfinity".to_string()));
      }
      return Ok(unevaluated("HarmonicNumber", args));
    }
    None => {
      return Ok(unevaluated("HarmonicNumber", args));
    }
  };

  let r = if args.len() == 2 {
    match expr_to_i128(&args[1]) {
      Some(r) => r,
      None => {
        return Ok(unevaluated("HarmonicNumber", args));
      }
    }
  } else {
    1
  };

  if n == 0 {
    return Ok(Expr::Integer(0));
  }

  // Compute as exact rational: sum of 1/k^r for k = 1 to n
  // Use BigInt numerator and denominator
  let mut num = BigInt::from(0);
  let mut den = BigInt::from(1);
  if r >= 0 {
    // Standard case: sum of 1/k^r
    for k in 1..=n {
      let k_big = BigInt::from(k);
      let k_pow = num_traits::pow::pow(k_big, r as usize);
      // Add 1/k_pow to num/den: num/den + 1/k_pow = (num*k_pow + den) / (den*k_pow)
      num = &num * &k_pow + &den;
      den = &den * &k_pow;
      // Reduce
      let g = gcd_bigint(&num, &den);
      use num_traits::One;
      if g > BigInt::one() {
        num /= &g;
        den /= &g;
      }
    }
  } else {
    // Negative r: sum of k^|r| (each term is an integer, result is integer)
    let abs_r = r.unsigned_abs() as usize;
    for k in 1..=n {
      let k_big = BigInt::from(k);
      let k_pow = num_traits::pow::pow(k_big, abs_r);
      num += k_pow;
    }
  }

  // Convert to our representation
  if den == BigInt::from(1) {
    Ok(bigint_to_expr(num))
  } else {
    Ok(Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![bigint_to_expr(num), bigint_to_expr(den)].into(),
    })
  }
}

/// MultipleHarmonicNumber[n, {s1, …, sk}] - the finite multiple harmonic sum
///   Sum over  n ≥ i1 > i2 > … > ik ≥ 1  of  Product_j 1/ij^sj
/// (strictly decreasing indices). The one-argument form
/// MultipleHarmonicNumber[n] uses the weight {1}, i.e. the ordinary harmonic
/// number H_n. Only a non-negative integer `n` and a list of positive-integer
/// weights are evaluated; anything else stays unevaluated.
pub fn multiple_harmonic_number_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("MultipleHarmonicNumber", args);
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated());
  }
  // `n` must be a non-negative integer.
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n,
    _ => return Ok(unevaluated()),
  };
  // Weight list: {1} for the one-argument form, otherwise a list of
  // positive integers.
  let exps: Vec<u32> = if args.len() == 1 {
    vec![1]
  } else if let Expr::List(items) = &args[1] {
    if items.is_empty() {
      return Ok(unevaluated());
    }
    let mut v = Vec::with_capacity(items.len());
    for it in items.iter() {
      match expr_to_i128(it) {
        Some(s) if s >= 1 && s <= i128::from(u32::MAX) => v.push(s as u32),
        _ => return Ok(unevaluated()),
      }
    }
    v
  } else {
    // A non-list second argument is a usage error in wolframscript.
    crate::emit_message(&format!(
      "MultipleHarmonicNumber::list: List expected at position 2 in {}.",
      expr_to_string(&unevaluated())
    ));
    return Ok(unevaluated());
  };

  // Recursively enumerate each strictly-decreasing index tuple, accumulating
  // its denominator ∏ ij^sj as a `1/denom` term.
  fn gen_terms(
    exps: &[u32],
    max_i: i128,
    denom: &BigInt,
    terms: &mut Vec<Expr>,
  ) {
    let Some((&s, rest)) = exps.split_first() else {
      terms.push(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(bigint_to_expr(denom.clone())),
      });
      return;
    };
    // Leave room for the `rest.len()` smaller indices still to be chosen.
    let mut i = rest.len() as i128 + 1;
    while i <= max_i {
      let factor = BigInt::from(i).pow(s);
      gen_terms(rest, i - 1, &(denom * &factor), terms);
      i += 1;
    }
  }
  let mut terms: Vec<Expr> = Vec::new();
  gen_terms(&exps, n, &BigInt::from(1), &mut terms);
  if terms.is_empty() {
    return Ok(Expr::Integer(0));
  }
  crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.into(),
  })
}

/// AlternatingHarmonicNumber[n] - Sum[(-1)^(k+1)/k, {k,1,n}].
/// AlternatingHarmonicNumber[n, r] - Sum[(-1)^(k+1)/k^r, {k,1,n}].
/// AlternatingHarmonicNumber[n, r, x] - Sum[(-1)^(k+1) x^k/k^r, {k,1,n}].
///
/// Only exact arguments are evaluated: wolframscript routes Real n/r through
/// its internal Hurwitz-zeta closed form (and an analytic continuation for
/// non-integer n), whose float rounding differs from direct summation in the
/// last digits, so Real n/r stay unevaluated here.
pub fn alternating_harmonic_number_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = unevaluated("AlternatingHarmonicNumber", args);

  // Exact negation of the order r, used to build k^(-r) terms.
  fn negate_exact(r: &Expr) -> Option<Expr> {
    match r {
      Expr::Integer(p) => Some(Expr::Integer(-p)),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        let p = expr_to_bigint(&args[0])?;
        Some(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![bigint_to_expr(-p), args[1].clone()].into(),
        })
      }
      _ => None,
    }
  }

  // Sum[(-1)^(k+1) x^k / k^r] built term by term; Woxi's Plus fold keeps
  // wolframscript's term order (e.g. 1 - 1/Sqrt[2] + 1/Sqrt[3]) and handles
  // exact, Real, and complex x uniformly.
  fn build_alternating_sum(n: i128, neg_r: &Expr, x: Option<&Expr>) -> Expr {
    let mut terms = Vec::new();
    for k in 1..=n {
      let k_pow = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::Integer(k)),
        right: Box::new(neg_r.clone()),
      };
      let mut factors = Vec::new();
      if k % 2 == 0 {
        factors.push(Expr::Integer(-1));
      }
      if let Some(x) = x {
        factors.push(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(x.clone()),
          right: Box::new(Expr::Integer(k)),
        });
      }
      factors.push(k_pow);
      terms.push(if factors.len() == 1 {
        factors.pop().unwrap()
      } else {
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: factors.into(),
        }
      });
    }
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    }
  }

  let is_exact_order = |r: &Expr| {
    expr_to_i128(r).is_some()
      || matches!(r, Expr::FunctionCall { name, args }
           if name == "Rational" && args.len() == 2)
  };

  // n = Infinity: closed forms via DirichletEta / PolyLog.
  let is_infinity = matches!(&args[0],
    Expr::Identifier(n) | Expr::Constant(n) if n == "Infinity");
  if is_infinity {
    let log2 = Expr::FunctionCall {
      name: "Log".to_string(),
      args: vec![Expr::Integer(2)].into(),
    };
    match args.len() {
      1 => return crate::evaluator::evaluate_expr_to_expr(&log2),
      2 => {
        let r = &args[1];
        if matches!(r, Expr::Real(_) | Expr::BigFloat(_, _)) {
          return Ok(unevaluated);
        }
        if expr_to_i128(r) == Some(1) {
          return crate::evaluator::evaluate_expr_to_expr(&log2);
        }
        // Eta[r] in wolframscript's form: ((-2 + 2^r) Zeta[r])/2^r
        let two_pow_r = |exp: Expr| Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(Expr::Integer(2)),
          right: Box::new(exp),
        };
        let eta = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![Expr::Integer(-2), two_pow_r(r.clone())].into(),
            },
            Expr::FunctionCall {
              name: "Zeta".to_string(),
              args: vec![r.clone()].into(),
            },
            two_pow_r(Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(-1), r.clone()].into(),
            }),
          ]
          .into(),
        };
        return crate::evaluator::evaluate_expr_to_expr(&eta);
      }
      _ => {
        // Sum[(-1)^(k+1) x^k/k^r, {k,1,Infinity}] = -PolyLog[r, -x]
        let poly_log = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Integer(-1),
            Expr::FunctionCall {
              name: "PolyLog".to_string(),
              args: vec![
                args[1].clone(),
                Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: vec![Expr::Integer(-1), args[2].clone()].into(),
                },
              ]
              .into(),
            },
          ]
          .into(),
        };
        return crate::evaluator::evaluate_expr_to_expr(&poly_log);
      }
    }
  }

  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n,
    // Negative integers and all non-exact-integer arguments stay symbolic
    // (wolframscript leaves AlternatingHarmonicNumber[-3] unevaluated too).
    _ => return Ok(unevaluated),
  };

  if args.len() <= 2 {
    // The empty and single-term sums collapse for ANY order, even symbolic:
    // wolframscript gives AlternatingHarmonicNumber[0, x] == 0 and [1, x] == 1.
    if n == 0 {
      return Ok(Expr::Integer(0));
    }
    if n == 1 {
      return Ok(Expr::Integer(1));
    }

    let r_expr = if args.len() == 2 {
      args[1].clone()
    } else {
      Expr::Integer(1)
    };

    if let Some(r) = expr_to_i128(&r_expr) {
      // Exact rational alternating sum, mirroring harmonic_number_ast.
      let mut num = BigInt::from(0);
      let mut den = BigInt::from(1);
      if r >= 0 {
        for k in 1..=n {
          let k_pow = num_traits::pow::pow(BigInt::from(k), r as usize);
          let signed = if k % 2 == 0 {
            -den.clone()
          } else {
            den.clone()
          };
          num = &num * &k_pow + signed;
          den = &den * &k_pow;
          let g = gcd_bigint(&num, &den);
          use num_traits::One;
          if g > BigInt::one() {
            num /= &g;
            den /= &g;
          }
        }
      } else {
        // Negative order: alternating power sum, always an integer.
        for k in 1..=n {
          let k_pow =
            num_traits::pow::pow(BigInt::from(k), r.unsigned_abs() as usize);
          if k % 2 == 0 {
            num -= k_pow;
          } else {
            num += k_pow;
          }
        }
      }
      return if den == BigInt::from(1) {
        Ok(bigint_to_expr(num))
      } else {
        Ok(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![bigint_to_expr(num), bigint_to_expr(den)].into(),
        })
      };
    }

    // Rational order: term-built symbolic sum (1 - 1/Sqrt[2] + 1/Sqrt[3]).
    if let Some(neg_r) = negate_exact(&r_expr) {
      let sum = build_alternating_sum(n, &neg_r, None);
      return crate::evaluator::evaluate_expr_to_expr(&sum);
    }
    return Ok(unevaluated);
  }

  // Three arguments: wolframscript only evaluates for concrete numeric x
  // (AlternatingHarmonicNumber[1, r, x] stays unevaluated).
  let r_expr = &args[1];
  let x = &args[2];
  if !is_exact_order(r_expr)
    || !crate::functions::predicate_ast::is_numeric_q(x)
  {
    return Ok(unevaluated);
  }
  if n == 0 {
    return Ok(Expr::Integer(0));
  }
  let neg_r = match negate_exact(r_expr) {
    Some(neg_r) => neg_r,
    None => return Ok(unevaluated),
  };
  let sum = build_alternating_sum(n, &neg_r, Some(x));
  crate::evaluator::evaluate_expr_to_expr(&sum)
}

/// HyperHarmonicNumber[r, n] - the hyperharmonic number H_n^(r): r-fold
/// iterated partial sums of the harmonic series.
/// HyperHarmonicNumber[r, n, s] - iterated sums over the base 1/k^s.
/// HyperHarmonicNumber[r, n, s, x] - iterated sums over the base x^k/k^s.
///
/// Evaluated via the coefficient form
/// Sum[Binomial[n - k + r - 1, r - 1] x^k / k^s, {k, 1, n}].
///
/// NOTE the argument order: the ORDER r comes first (wolframscript's
/// convention), so HyperHarmonicNumber[1, n] == HarmonicNumber[n].
///
/// Only exact arguments are evaluated: wolframscript computes Real arguments
/// through digamma/Gamma closed forms whose float rounding differs from
/// direct summation in the last digits, so Reals stay unevaluated here.
/// A symbolic s with r >= 1 and n >= 1 also stays unevaluated — wolframscript
/// leaks the internal System`HarmonicNumberDump`MQHN[...] symbol for it,
/// which is a bug not worth reproducing.
pub fn hyper_harmonic_number_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = unevaluated("HyperHarmonicNumber", args);

  let contains_real = |e: &Expr| -> bool {
    fn walk(e: &Expr) -> bool {
      match e {
        Expr::Real(_) | Expr::BigFloat(_, _) => true,
        Expr::BinaryOp { left, right, .. } => walk(left) || walk(right),
        Expr::UnaryOp { operand, .. } => walk(operand),
        Expr::FunctionCall { args, .. } | Expr::List(args) => {
          args.iter().any(walk)
        }
        _ => false,
      }
    }
    walk(e)
  };
  if args.iter().any(contains_real) {
    return Ok(unevaluated);
  }

  // The order r and the index n must be concrete non-negative integers.
  let (Some(r), Some(n)) = (expr_to_i128(&args[0]), expr_to_i128(&args[1]))
  else {
    return Ok(unevaluated);
  };
  if r < 0 || n < 0 {
    return Ok(unevaluated);
  }

  let s = if args.len() >= 3 {
    args[2].clone()
  } else {
    Expr::Integer(1)
  };
  let x = args.get(3);

  // The empty sum is 0 for any s and x, even symbolic ones.
  if n == 0 {
    return Ok(Expr::Integer(0));
  }

  let neg_s = |s: &Expr| -> Expr {
    match s {
      Expr::Integer(p) => Expr::Integer(-p),
      _ => Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), s.clone()].into(),
      },
    }
  };

  // r = 0 is the plain base term x^n / n^s; wolframscript evaluates it even
  // for symbolic s and x (e.g. HyperHarmonicNumber[0, 5, s] -> 5^(-s)).
  if r == 0 {
    let n_pow = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(Expr::Integer(n)),
      right: Box::new(neg_s(&s)),
    };
    let term = match x {
      Some(x) => Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(x.clone()),
            right: Box::new(Expr::Integer(n)),
          },
          n_pow,
        ]
        .into(),
      },
      None => n_pow,
    };
    return crate::evaluator::evaluate_expr_to_expr(&term);
  }

  // r >= 1: the exponent s must be an exact Integer/Rational, and x (when
  // present) a concrete numeric value.
  let s_is_exact = expr_to_i128(&s).is_some()
    || matches!(&s, Expr::FunctionCall { name, args }
         if name == "Rational" && args.len() == 2);
  if !s_is_exact {
    return Ok(unevaluated);
  }
  if let Some(x) = x
    && !crate::functions::predicate_ast::is_numeric_q(x)
  {
    return Ok(unevaluated);
  }

  // Binomial[n - k + r - 1, r - 1] as an exact BigInt.
  let binom = |m: i128, j: i128| -> BigInt {
    let mut acc = BigInt::from(1);
    for i in 0..j {
      acc = acc * BigInt::from(m - i) / BigInt::from(i + 1);
    }
    acc
  };

  let neg_s = neg_s(&s);
  let mut terms = Vec::new();
  for k in 1..=n {
    let coeff = bigint_to_expr(binom(n - k + r - 1, r - 1));
    let k_pow = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(Expr::Integer(k)),
      right: Box::new(neg_s.clone()),
    };
    let mut factors = vec![coeff];
    if let Some(x) = x {
      factors.push(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(x.clone()),
        right: Box::new(Expr::Integer(k)),
      });
    }
    factors.push(k_pow);
    terms.push(Expr::FunctionCall {
      name: "Times".to_string(),
      args: factors.into(),
    });
  }
  let sum = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&sum)
}

/// Prime[n] - Returns the nth prime number
pub fn prime_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Prime expects exactly 1 argument".into(),
    ));
  }
  let unevaluated = unevaluated("Prime", args);
  // Prime requires an exact positive integer index. wolframscript rejects
  // every Real argument with Prime::intpp, even an integer-valued one like
  // 3.0, so do NOT coerce Reals to integers here.
  match expr_to_i128(&args[0]) {
    Some(n) if n >= 1 => Ok(Expr::Integer(nth_prime(n))),
    Some(_) => {
      // Concrete non-positive integer: emit message like wolframscript
      let arg_str = expr_to_string(&args[0]);
      crate::emit_message(&format!(
        "Prime::intpp: Positive integer argument expected in Prime[{}].",
        arg_str
      ));
      Ok(unevaluated)
    }
    None => {
      // Concrete non-integer numeric (e.g. 1.5): emit message like wolframscript
      // Symbolic arguments return unevaluated silently
      if matches!(&args[0], Expr::Real(_) | Expr::BigFloat(_, _)) {
        let arg_str = expr_to_string(&args[0]);
        crate::emit_message(&format!(
          "Prime::intpp: Positive integer argument expected in Prime[{}].",
          arg_str
        ));
      }
      Ok(unevaluated)
    }
  }
}

/// Fibonacci[n] - Returns the nth Fibonacci number.
/// Fibonacci[n, x] - Returns the nth Fibonacci polynomial in x.
///
/// The Fibonacci polynomials satisfy F_0(x) = 0, F_1(x) = 1, and
/// F_n(x) = x * F_{n-1}(x) + F_{n-2}(x). For negative indices,
/// F_{-n}(x) = (-1)^{n+1} F_n(x). When `x` is itself an integer or
/// rational, the recurrence collapses to a single number.
pub fn fibonacci_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 2 {
    return fibonacci_polynomial_ast(&args[0], &args[1]);
  }

  // Real argument: evaluate via the analytic continuation
  // Fibonacci[x] = (phi^x - Cos[pi x] phi^-x) / Sqrt[5], where phi is the
  // golden ratio. This is what wolframscript returns for non-integer (and
  // for N[Fibonacci[1/2]], which first numericizes the index to a Real).
  if let Expr::Real(x) = &args[0] {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let val = (phi.powf(*x)
      - (std::f64::consts::PI * *x).cos() * phi.powf(-*x))
      / 5.0_f64.sqrt();
    return Ok(Expr::Real(val));
  }

  match expr_to_i128(&args[0]) {
    Some(n) => {
      if n < 0 {
        let pos_n = (-n) as u128;
        let fib = fibonacci_number_bigint(pos_n);
        let sign = if pos_n.is_multiple_of(2) {
          BigInt::from(-1)
        } else {
          BigInt::from(1)
        };
        Ok(bigint_to_expr(sign * fib))
      } else {
        Ok(bigint_to_expr(fibonacci_number_bigint(n as u128)))
      }
    }
    None => Ok(unevaluated("Fibonacci", args)),
  }
}

/// Fibonacci[n, x] - the n-th Fibonacci polynomial evaluated at `x`.
fn fibonacci_polynomial_ast(
  n_arg: &Expr,
  x: &Expr,
) -> Result<Expr, InterpreterError> {
  let Some(n) = expr_to_i128(n_arg) else {
    return Ok(Expr::FunctionCall {
      name: "Fibonacci".to_string(),
      args: vec![n_arg.clone(), x.clone()].into(),
    });
  };

  // Negative index: F_{-n}(x) = (-1)^{n+1} F_n(x)
  if n < 0 {
    let pos = -n;
    let positive = fibonacci_polynomial_ast(&Expr::Integer(pos), x)?;
    if pos % 2 == 0 {
      // (-1)^{n+1} = -1 when n is even (since -n flips parity to odd)
      return crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[Expr::Integer(-1), positive],
      );
    }
    return Ok(positive);
  }

  if n == 0 {
    return Ok(Expr::Integer(0));
  }
  if n == 1 {
    return Ok(Expr::Integer(1));
  }

  // Iterative recurrence: a = F_{k-2}, b = F_{k-1}, next = x*b + a.
  // We let the regular evaluator simplify each step so that integer x
  // collapses to a number while symbolic x builds up the polynomial.
  let mut a = Expr::Integer(0);
  let mut b = Expr::Integer(1);
  for _ in 2..=n {
    let xb = crate::evaluator::evaluate_function_call_ast(
      "Times",
      &[x.clone(), b.clone()],
    )?;
    let next = crate::evaluator::evaluate_function_call_ast("Plus", &[xb, a])?;
    a = b;
    b = next;
  }
  // For symbolic `x`, expand the result so it prints in canonical form.
  if matches!(x, Expr::Integer(_) | Expr::BigInteger(_) | Expr::Real(_)) {
    Ok(b)
  } else {
    crate::evaluator::evaluate_function_call_ast("Expand", &[b])
  }
}

fn fibonacci_number_bigint(n: u128) -> BigInt {
  if n == 0 {
    return BigInt::from(0);
  }
  let mut a = BigInt::from(0);
  let mut b = BigInt::from(1);
  for _ in 1..n {
    let tmp = &a + &b;
    a = b;
    b = tmp;
  }
  b
}

/// FactorInteger[n] - Returns the prime factorization of n
pub fn factor_integer_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "FactorInteger expects 1 or 2 arguments".into(),
    ));
  }

  // FactorInteger[n, GaussianIntegers -> True/False]: factor over the
  // Gaussian integers. `False` behaves like the ordinary 1-argument form.
  if args.len() == 2
    && let Expr::Rule {
      pattern,
      replacement,
    } = &args[1]
    && matches!(pattern.as_ref(), Expr::Identifier(s) if s == "GaussianIntegers")
  {
    return match replacement.as_ref() {
      Expr::Identifier(s) if s == "True" => factor_integer_gaussian(&args[0]),
      _ => factor_integer_ast(&args[..1]),
    };
  }

  // FactorInteger[n, k]: partial factorization, pulling out at most k distinct
  // factors.
  if args.len() == 2 {
    return factor_integer_partial(&args[0], &args[1]);
  }

  match &args[0] {
    Expr::Integer(n) => factor_integer_i128(*n),
    Expr::BigInteger(n) => factor_integer_bigint(n),
    // Handle Rational[numerator, denominator]
    Expr::FunctionCall {
      name,
      args: rat_args,
    } if name == "Rational" && rat_args.len() == 2 => {
      factor_integer_rational(&rat_args[0], &rat_args[1])
    }
    // A Gaussian integer argument factors over ℤ[i] directly.
    e if extract_gaussian_integer(e).is_some_and(|(_, im)| im != 0) => {
      factor_integer_gaussian(&args[0])
    }
    _ => Ok(unevaluated("FactorInteger", args)),
  }
}

/// Recognize an exact Gaussian integer `re + im I`.
fn extract_gaussian_integer(expr: &Expr) -> Option<(i128, i128)> {
  if let Expr::Integer(n) = expr {
    return Some((*n, 0));
  }
  let ((rn, rd), (in_, id)) =
    crate::functions::math_ast::try_extract_complex_exact(expr)?;
  if rd == 1 && id == 1 {
    Some((rn, in_))
  } else {
    None
  }
}

/// `FactorInteger[n, GaussianIntegers -> True]` — factorization over the
/// Gaussian integers ℤ[i]. The result is `{{unit, 1}, {prime, exp}, …}` where
/// the optional leading unit (one of ±1, ±I) is shown only when it is not 1,
/// and the Gaussian primes are sorted by norm then by real part.
///
/// Each rational prime `p` of `|n|` lifts as:
///   * `p = 2`:        the ramified prime `(1 + I)`, exponent doubled,
///                     contributing a unit factor `(-I)^e`.
///   * `p ≡ 1 (mod 4)`: splits into conjugates `(a + b I)` and `(b + a I)`
///                     where `a² + b² = p`, `0 < a < b`, contributing `(-I)^e`.
///   * `p ≡ 3 (mod 4)`: inert — the rational prime `p` itself, unit unchanged.
/// A negative `n` contributes an extra `-1` to the unit.
fn factor_integer_gaussian(n_expr: &Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "FactorInteger".to_string(),
      args: vec![
        n_expr.clone(),
        Expr::Rule {
          pattern: Box::new(Expr::Identifier("GaussianIntegers".to_string())),
          replacement: Box::new(bool_expr(true)),
        },
      ]
      .into(),
    })
  };

  let Some((n, n_im)) = extract_gaussian_integer(n_expr) else {
    return unevaluated();
  };
  if n_im != 0 {
    return factor_gaussian_complex(n, n_im, &unevaluated);
  }

  let single = |re: i128, im: i128| -> Result<Expr, InterpreterError> {
    Ok(Expr::List(
      vec![Expr::List(
        vec![gaussian_to_expr(re, im)?, Expr::Integer(1)].into(),
      )]
      .into(),
    ))
  };
  // Units and zero are returned as a single {value, 1} entry.
  if n == 0 || n == 1 || n == -1 {
    return single(n, 0);
  }

  let mut num = n.unsigned_abs();
  // Restrict to the trial-division-friendly range; larger inputs would need
  // the BigInt path and are left unevaluated.
  if num > (1u128 << 53) {
    return unevaluated();
  }

  // Factor |n| into rational primes (prime, exponent).
  let mut rational: Vec<(i128, i128)> = Vec::new();
  let mut c2 = 0i128;
  while num.is_multiple_of(2) {
    c2 += 1;
    num /= 2;
  }
  if c2 > 0 {
    rational.push((2, c2));
  }
  let mut i: u128 = 3;
  while i * i <= num {
    let mut c = 0i128;
    while num.is_multiple_of(i) {
      c += 1;
      num /= i;
    }
    if c > 0 {
      rational.push((i as i128, c));
    }
    i += 2;
  }
  if num > 1 {
    rational.push((num as i128, 1));
  }

  // Lift each rational prime, accumulating the unit (re, im) ∈ {±1, ±i}.
  // Multiplication by -i sends (re, im) → (im, -re).
  let mut unit = (if n < 0 { -1i128 } else { 1i128 }, 0i128);
  let mut gprimes: Vec<((i128, i128), i128)> = Vec::new();
  for (p, e) in rational {
    if p == 2 {
      gprimes.push(((1, 1), 2 * e));
      for _ in 0..e {
        unit = (unit.1, -unit.0);
      }
    } else if p % 4 == 1 {
      let Some((a, b)) = sum_of_two_squares(p) else {
        return unevaluated();
      };
      gprimes.push(((a, b), e));
      gprimes.push(((b, a), e));
      for _ in 0..e {
        unit = (unit.1, -unit.0);
      }
    } else {
      gprimes.push(((p, 0), e));
    }
  }

  // Sort by real part, then by imaginary part (wolframscript's order: e.g.
  // the inert prime 3 sorts between 2 + 3 I and 3 + 2 I).
  gprimes.sort_by_key(|&((re, im), _)| (re, im));

  let mut factors: Vec<Expr> = Vec::new();
  if unit != (1, 0) {
    factors.push(Expr::List(
      vec![gaussian_to_expr(unit.0, unit.1)?, Expr::Integer(1)].into(),
    ));
  }
  for ((re, im), e) in gprimes {
    factors.push(Expr::List(
      vec![gaussian_to_expr(re, im)?, Expr::Integer(e)].into(),
    ));
  }
  Ok(Expr::List(factors.into()))
}

/// Exact quotient z/w in ℤ[i], or None when w does not divide z.
fn gaussian_div(z: (i128, i128), w: (i128, i128)) -> Option<(i128, i128)> {
  let n = w.0 * w.0 + w.1 * w.1;
  let re = z.0 * w.0 + z.1 * w.1;
  let im = z.1 * w.0 - z.0 * w.1;
  if re % n == 0 && im % n == 0 {
    Some((re / n, im / n))
  } else {
    None
  }
}

/// Factorization of a proper Gaussian integer `a + b I` (b != 0): factor
/// the norm a² + b² into rational primes, then trial-divide by the
/// canonical prime representatives (first quadrant: 1 + I for 2, the
/// {s + t I, t + s I} pair for split primes, the rational prime itself
/// for inert ones); whatever remains is the leading unit.
fn factor_gaussian_complex(
  a: i128,
  b: i128,
  unevaluated: &dyn Fn() -> Result<Expr, InterpreterError>,
) -> Result<Expr, InterpreterError> {
  let Some(norm) = a
    .checked_mul(a)
    .and_then(|x| b.checked_mul(b).and_then(|y| x.checked_add(y)))
    .filter(|&n| n <= (1i128 << 53))
  else {
    return unevaluated();
  };

  // Factor the norm into rational primes.
  let mut num = norm as u128;
  let mut rational: Vec<(i128, i128)> = Vec::new();
  let mut c2 = 0i128;
  while num.is_multiple_of(2) {
    c2 += 1;
    num /= 2;
  }
  if c2 > 0 {
    rational.push((2, c2));
  }
  let mut i: u128 = 3;
  while i * i <= num {
    let mut c = 0i128;
    while num.is_multiple_of(i) {
      c += 1;
      num /= i;
    }
    if c > 0 {
      rational.push((i as i128, c));
    }
    i += 2;
  }
  if num > 1 {
    rational.push((num as i128, 1));
  }

  let mut z = (a, b);
  let mut gprimes: Vec<((i128, i128), i128)> = Vec::new();
  for (p, e) in rational {
    if p == 2 {
      // Each factor 2 of the norm is one ramified prime 1 + I in z.
      for _ in 0..e {
        let Some(w) = gaussian_div(z, (1, 1)) else {
          return unevaluated();
        };
        z = w;
      }
      gprimes.push(((1, 1), e));
    } else if p % 4 == 3 {
      // Inert: the norm exponent is twice the multiplicity in z.
      if e % 2 != 0 {
        return unevaluated();
      }
      for _ in 0..(e / 2) {
        let Some(w) = gaussian_div(z, (p, 0)) else {
          return unevaluated();
        };
        z = w;
      }
      gprimes.push(((p, 0), e / 2));
    } else {
      let Some((s, t)) = sum_of_two_squares(p) else {
        return unevaluated();
      };
      let mut k1 = 0i128;
      while k1 < e {
        let Some(w) = gaussian_div(z, (s, t)) else {
          break;
        };
        z = w;
        k1 += 1;
      }
      for _ in 0..(e - k1) {
        let Some(w) = gaussian_div(z, (t, s)) else {
          return unevaluated();
        };
        z = w;
      }
      if k1 > 0 {
        gprimes.push(((s, t), k1));
      }
      if e - k1 > 0 {
        gprimes.push(((t, s), e - k1));
      }
    }
  }

  // The fully divided z is one of the four units.
  gprimes.sort_by_key(|&((re, im), _)| (re, im));
  let mut factors: Vec<Expr> = Vec::new();
  if z != (1, 0) || gprimes.is_empty() {
    factors.push(Expr::List(
      vec![gaussian_to_expr(z.0, z.1)?, Expr::Integer(1)].into(),
    ));
  }
  for ((re, im), e) in gprimes {
    factors.push(Expr::List(
      vec![gaussian_to_expr(re, im)?, Expr::Integer(e)].into(),
    ));
  }
  Ok(Expr::List(factors.into()))
}

/// Canonical Gaussian factorization of a nonzero Gaussian integer:
/// (unit, [(first-quadrant prime, exponent)]) sorted by (re, im). Uses the
/// same norm-factorization + trial-division construction as
/// `factor_gaussian_complex` (with which it agrees for real inputs too).
fn gaussian_factorization(
  a: i128,
  b: i128,
) -> Option<((i128, i128), Vec<((i128, i128), i128)>)> {
  let norm = a
    .checked_mul(a)
    .and_then(|x| b.checked_mul(b).and_then(|y| x.checked_add(y)))
    .filter(|&n| n > 0 && n <= (1i128 << 53))?;

  let mut num = norm as u128;
  let mut rational: Vec<(i128, i128)> = Vec::new();
  let mut c2 = 0i128;
  while num.is_multiple_of(2) {
    c2 += 1;
    num /= 2;
  }
  if c2 > 0 {
    rational.push((2, c2));
  }
  let mut i: u128 = 3;
  while i * i <= num {
    let mut c = 0i128;
    while num.is_multiple_of(i) {
      c += 1;
      num /= i;
    }
    if c > 0 {
      rational.push((i as i128, c));
    }
    i += 2;
  }
  if num > 1 {
    rational.push((num as i128, 1));
  }

  let mut z = (a, b);
  let mut gprimes: Vec<((i128, i128), i128)> = Vec::new();
  for (p, e) in rational {
    if p == 2 {
      for _ in 0..e {
        z = gaussian_div(z, (1, 1))?;
      }
      gprimes.push(((1, 1), e));
    } else if p % 4 == 3 {
      if e % 2 != 0 {
        return None;
      }
      for _ in 0..(e / 2) {
        z = gaussian_div(z, (p, 0))?;
      }
      gprimes.push(((p, 0), e / 2));
    } else {
      let (s, t) = sum_of_two_squares(p)?;
      let mut k1 = 0i128;
      while k1 < e {
        let Some(w) = gaussian_div(z, (s, t)) else {
          break;
        };
        z = w;
        k1 += 1;
      }
      for _ in 0..(e - k1) {
        z = gaussian_div(z, (t, s))?;
      }
      if k1 > 0 {
        gprimes.push(((s, t), k1));
      }
      if e - k1 > 0 {
        gprimes.push(((t, s), e - k1));
      }
    }
  }
  gprimes.sort_by_key(|&((re, im), _)| (re, im));
  Some((z, gprimes))
}

fn gaussian_mul_checked(
  a: (i128, i128),
  b: (i128, i128),
) -> Option<(i128, i128)> {
  let re = a.0.checked_mul(b.0)?.checked_sub(a.1.checked_mul(b.1)?)?;
  let im = a.0.checked_mul(b.1)?.checked_add(a.1.checked_mul(b.0)?)?;
  Some((re, im))
}

fn gaussian_pow(mut base: (i128, i128), mut e: i128) -> Option<(i128, i128)> {
  let mut acc = (1i128, 0i128);
  while e > 0 {
    if e & 1 == 1 {
      acc = gaussian_mul_checked(acc, base)?;
    }
    e >>= 1;
    if e > 0 {
      base = gaussian_mul_checked(base, base)?;
    }
  }
  Some(acc)
}

/// The first-quadrant associate (re > 0, im >= 0) of a nonzero Gaussian
/// integer.
fn gaussian_first_quadrant(mut z: (i128, i128)) -> (i128, i128) {
  for _ in 0..3 {
    if z.0 > 0 && z.1 >= 0 {
      return z;
    }
    z = (-z.1, z.0); // multiply by I
  }
  z
}

/// Divisors of a Gaussian integer (or of a rational integer over ℤ[i]):
/// one first-quadrant representative per associate class, sorted by
/// (re, im). None for zero, overflow, or too many divisors.
fn gaussian_divisors(a: i128, b: i128) -> Option<Vec<(i128, i128)>> {
  let (_, gprimes) = gaussian_factorization(a, b)?;
  let count: i128 = gprimes.iter().map(|&(_, e)| e + 1).product();
  if count > 4096 {
    return None;
  }
  let mut divisors: Vec<(i128, i128)> = vec![(1, 0)];
  for &(p, e) in &gprimes {
    let mut next = Vec::with_capacity(divisors.len() * (e as usize + 1));
    for d in &divisors {
      let mut acc = *d;
      next.push(acc);
      for _ in 0..e {
        acc = gaussian_mul_checked(acc, p)?;
        next.push(acc);
      }
    }
    divisors = next;
  }
  let mut out: Vec<(i128, i128)> =
    divisors.into_iter().map(gaussian_first_quadrant).collect();
  out.sort();
  Some(out)
}

/// DivisorSigma over ℤ[i]: k = 0 counts associate classes; k >= 1 uses the
/// multiplicative formula prod (p^(k(e+1)) - 1)/(p^k - 1) over the
/// canonical first-quadrant primes.
fn gaussian_divisor_sigma(k: i128, a: i128, b: i128) -> Option<(i128, i128)> {
  let (_, gprimes) = gaussian_factorization(a, b)?;
  if k == 0 {
    let count: i128 = gprimes.iter().map(|&(_, e)| e + 1).product();
    return Some((count, 0));
  }
  let mut acc = (1i128, 0i128);
  for &(p, e) in &gprimes {
    let num = {
      let (re, im) = gaussian_pow(p, k.checked_mul(e + 1)?)?;
      (re - 1, im)
    };
    let den = {
      let (re, im) = gaussian_pow(p, k)?;
      (re - 1, im)
    };
    let term = gaussian_div(num, den)?;
    acc = gaussian_mul_checked(acc, term)?;
  }
  Some(acc)
}

/// Render a Gaussian integer `re + im·I` as an `Expr` — a bare `Integer` when
/// `im == 0`, otherwise the evaluated `Complex[re, im]`.
fn gaussian_to_expr(re: i128, im: i128) -> Result<Expr, InterpreterError> {
  if im == 0 {
    return Ok(Expr::Integer(re));
  }
  crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Complex".to_string(),
    args: vec![Expr::Integer(re), Expr::Integer(im)].into(),
  })
}

/// For a prime `p ≡ 1 (mod 4)`, return `(a, b)` with `a² + b² = p` and
/// `0 < a < b` (unique up to order by Fermat's two-square theorem).
fn sum_of_two_squares(p: i128) -> Option<(i128, i128)> {
  let mut a = 1i128;
  while a * a * 2 < p {
    let r = p - a * a;
    // Integer square root of r, robust against f64 rounding.
    let mut b = (r as f64).sqrt() as i128;
    while b * b > r {
      b -= 1;
    }
    while (b + 1) * (b + 1) <= r {
      b += 1;
    }
    if b * b == r && b > a {
      return Some((a, b));
    }
    a += 1;
  }
  None
}

/// `FactorInteger[n, k]` — partial factorization that pulls out at most `k`
/// distinct factors. The full factorization is collapsed by keeping the `k - 1`
/// largest prime powers separate and combining the remaining (smaller) prime
/// powers into a single composite cofactor, e.g. `FactorInteger[60, 2]` →
/// `{{5, 1}, {12, 1}}`. `k = 1` returns `{{n, 1}}` (no factoring). When the
/// number already has `≤ k` distinct primes the full factorization is returned.
///
/// Only non-negative `n` is handled: for negative `n` wolframscript's choice of
/// which prime the sign attaches to is internal to its factoring algorithm and
/// not reproducible, so that case is left unevaluated.
fn factor_integer_partial(
  n_expr: &Expr,
  k_expr: &Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "FactorInteger".to_string(),
      args: vec![n_expr.clone(), k_expr.clone()].into(),
    })
  };

  // k must be a positive integer.
  let k = match k_expr {
    Expr::Integer(k) if *k >= 1 => *k,
    _ => return unevaluated(),
  };

  // n must be a non-negative machine integer.
  let n = match n_expr {
    Expr::Integer(n) if *n >= 0 => *n,
    _ => return unevaluated(),
  };

  // k == 1 never factors: report the number itself.
  if k == 1 {
    return Ok(Expr::List(
      vec![Expr::List(vec![Expr::Integer(n), Expr::Integer(1)].into())].into(),
    ));
  }

  // Fully factor, then extract the (prime, exponent) pairs (sorted ascending by
  // prime, which `factor_integer_i128` already guarantees).
  let full = factor_integer_i128(n)?;
  let pairs: Vec<(i128, i128)> = match &full {
    Expr::List(items) => items
      .iter()
      .filter_map(|it| match it {
        Expr::List(pv)
          if pv.len() == 2
            && matches!(pv[0], Expr::Integer(_))
            && matches!(pv[1], Expr::Integer(_)) =>
        {
          match (&pv[0], &pv[1]) {
            (Expr::Integer(p), Expr::Integer(e)) => Some((*p, *e)),
            _ => None,
          }
        }
        _ => None,
      })
      .collect(),
    _ => return unevaluated(),
  };

  let m = pairs.len() as i128;
  let result_pairs: Vec<(i128, i128)> = if m <= k {
    // Already few enough distinct primes — full factorization.
    pairs
  } else {
    // Combine the smallest (m - (k - 1)) prime powers into one cofactor and
    // keep the (k - 1) largest prime powers separate. The cofactor divides n,
    // so it cannot overflow i128.
    let n_combine = (m - (k - 1)) as usize;
    let mut cofactor: i128 = 1;
    for &(p, e) in &pairs[..n_combine] {
      cofactor *= p.pow(e as u32);
    }
    let mut out = vec![(cofactor, 1)];
    out.extend_from_slice(&pairs[n_combine..]);
    out.sort_by_key(|&(p, _)| p);
    out
  };

  Ok(Expr::List(
    result_pairs
      .into_iter()
      .map(|(p, e)| Expr::List(vec![Expr::Integer(p), Expr::Integer(e)].into()))
      .collect::<Vec<_>>()
      .into(),
  ))
}

/// Factor a rational number p/q: factor numerator and denominator separately,
/// then merge with denominator factors getting negative exponents.
fn factor_integer_rational(
  numer: &Expr,
  denom: &Expr,
) -> Result<Expr, InterpreterError> {
  use std::collections::BTreeMap;

  // Factor numerator and denominator
  let numer_factors = factor_integer_ast(&[numer.clone()])?;
  let denom_factors = factor_integer_ast(&[denom.clone()])?;

  // Collect into a map: prime -> exponent
  let mut factor_map: BTreeMap<i128, i128> = BTreeMap::new();
  let mut has_neg_one = false;

  if let Expr::List(pairs) = &numer_factors {
    for pair in pairs {
      if let Expr::List(pv) = pair
        && pv.len() == 2
        && let Expr::Integer(p) = &pv[0]
        && let Expr::Integer(e) = &pv[1]
      {
        if *p == -1 {
          has_neg_one = true;
        } else if *p != 1 {
          // Drop the trivial unit factor from FactorInteger[1] = {{1, 1}};
          // wolframscript: FactorInteger[1/6] = {{2, -1}, {3, -1}}.
          *factor_map.entry(*p).or_insert(0) += e;
        }
      }
    }
  }

  if let Expr::List(pairs) = &denom_factors {
    for pair in pairs {
      if let Expr::List(pv) = pair
        && pv.len() == 2
        && let Expr::Integer(p) = &pv[0]
        && let Expr::Integer(e) = &pv[1]
      {
        if *p == -1 {
          has_neg_one = !has_neg_one;
        } else if *p != 1 {
          *factor_map.entry(*p).or_insert(0) -= e;
        }
      }
    }
  }

  let mut result: Vec<Expr> = Vec::new();

  if has_neg_one {
    result.push(Expr::List(vec![Expr::Integer(-1), Expr::Integer(1)].into()));
  }

  for (prime, exp) in &factor_map {
    if *exp != 0 {
      result.push(Expr::List(
        vec![Expr::Integer(*prime), Expr::Integer(*exp)].into(),
      ));
    }
  }

  Ok(Expr::List(result.into()))
}

fn factor_integer_i128(n: i128) -> Result<Expr, InterpreterError> {
  if n == 0 {
    // wolframscript: FactorInteger[0] = {{0, 1}} (0 treated as 0^1).
    return Ok(Expr::List(
      vec![Expr::List(vec![Expr::Integer(0), Expr::Integer(1)].into())].into(),
    ));
  }

  // For large integers where trial division would be too slow,
  // delegate to the BigInt path which uses Pollard's rho
  if n.unsigned_abs() > (1u128 << 53) {
    return factor_integer_bigint(&BigInt::from(n));
  }

  let mut factors: Vec<Expr> = Vec::new();
  let mut num = n.unsigned_abs();

  if n < 0 {
    factors.push(Expr::List(vec![Expr::Integer(-1), Expr::Integer(1)].into()));
  }

  if num == 1 {
    if factors.is_empty() {
      // FactorInteger[1] → {{1, 1}}
      factors.push(Expr::List(vec![Expr::Integer(1), Expr::Integer(1)].into()));
    }
    return Ok(Expr::List(factors.into()));
  }

  // Handle factor of 2
  let mut count = 0i128;
  while num.is_multiple_of(2) {
    count += 1;
    num /= 2;
  }
  if count > 0 {
    factors.push(Expr::List(
      vec![Expr::Integer(2), Expr::Integer(count)].into(),
    ));
  }

  // Handle odd factors (safe for small n where trial division is fast)
  let mut i: u128 = 3;
  while i * i <= num {
    let mut count = 0i128;
    while num.is_multiple_of(i) {
      count += 1;
      num /= i;
    }
    if count > 0 {
      factors.push(Expr::List(
        vec![Expr::Integer(i as i128), Expr::Integer(count)].into(),
      ));
    }
    i += 2;
  }

  if num > 1 {
    factors.push(Expr::List(
      vec![Expr::Integer(num as i128), Expr::Integer(1)].into(),
    ));
  }

  Ok(Expr::List(factors.into()))
}

fn factor_integer_bigint(n: &BigInt) -> Result<Expr, InterpreterError> {
  use num_traits::{One, Signed, Zero};

  if n.is_zero() {
    // wolframscript: FactorInteger[0] = {{0, 1}} (0 treated as 0^1).
    return Ok(Expr::List(
      vec![Expr::List(vec![Expr::Integer(0), Expr::Integer(1)].into())].into(),
    ));
  }

  let mut factors: Vec<Expr> = Vec::new();

  if n.is_negative() {
    factors.push(Expr::List(vec![Expr::Integer(-1), Expr::Integer(1)].into()));
  }

  let mut remaining = n.abs();
  let one = BigInt::one();

  if remaining == one {
    return Ok(Expr::List(factors.into()));
  }

  // Trial division for small primes
  let two = BigInt::from(2);
  let mut count = 0i128;
  while (&remaining % &two).is_zero() {
    count += 1;
    remaining /= &two;
  }
  if count > 0 {
    factors.push(Expr::List(
      vec![Expr::Integer(2), Expr::Integer(count)].into(),
    ));
  }

  let trial_limit = 1_000_000u64;
  let mut i = 3u64;
  while i <= trial_limit {
    let bi = BigInt::from(i);
    if &bi * &bi > remaining {
      break;
    }
    let mut count = 0i128;
    while (&remaining % &bi).is_zero() {
      count += 1;
      remaining /= &bi;
    }
    if count > 0 {
      factors.push(Expr::List(
        vec![Expr::Integer(i as i128), Expr::Integer(count)].into(),
      ));
    }
    i += 2;
  }

  // Factor remaining cofactor using num_prime (Pollard's rho + SQUFOF)
  if remaining > one {
    let remaining_uint = remaining.to_biguint().unwrap();
    let prime_factors = num_prime::nt_funcs::factorize(remaining_uint);
    for (factor, exponent) in prime_factors {
      let factor_bigint = BigInt::from(factor);
      // Merge with existing factor or add new entry
      let mut merged = false;
      for f in factors.iter_mut() {
        if let Expr::List(pair) = f {
          let matches = match (&pair[0], &factor_bigint) {
            (Expr::Integer(a), b) => BigInt::from(*a) == *b,
            (Expr::BigInteger(a), b) => a == b,
            _ => false,
          };
          if matches {
            if let Expr::Integer(ref mut exp) = pair[1] {
              *exp += exponent as i128;
            }
            merged = true;
            break;
          }
        }
      }
      if !merged {
        factors.push(Expr::List(
          vec![
            bigint_to_expr(factor_bigint),
            Expr::Integer(exponent as i128),
          ]
          .into(),
        ));
      }
    }
  }

  // Sort factors by prime value
  factors.sort_by(|a, b| {
    let a_val = match a {
      Expr::List(pair) => match &pair[0] {
        Expr::Integer(n) => BigInt::from(*n),
        Expr::BigInteger(n) => n.clone(),
        _ => BigInt::zero(),
      },
      _ => BigInt::zero(),
    };
    let b_val = match b {
      Expr::List(pair) => match &pair[0] {
        Expr::Integer(n) => BigInt::from(*n),
        Expr::BigInteger(n) => n.clone(),
        _ => BigInt::zero(),
      },
      _ => BigInt::zero(),
    };
    a_val.cmp(&b_val)
  });

  Ok(Expr::List(factors.into()))
}

// ─── IntegerPartitions ─────────────────────────────────────────────
// IntegerPartitions[n] — all partitions of n
// IntegerPartitions[n, k] — partitions with at most k parts
// IntegerPartitions[n, {k}] — partitions with exactly k parts
// IntegerPartitions[n, {kmin, kmax}] — partitions with kmin..kmax parts
// IntegerPartitions[n, kspec, {d1, d2, ...}] — using only given elements
// kspec can be All (equivalent to no constraint)
pub fn integer_partitions_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Parse n
  let n = match expr_to_i128(&args[0]) {
    Some(v) => v,
    None => {
      return Ok(unevaluated("IntegerPartitions", args));
    }
  };

  if n < 0 {
    return Ok(Expr::List(vec![].into()));
  }
  let n = n as u64;

  // Positive infinity, either the symbol Infinity or DirectedInfinity[1].
  let is_pos_inf = |e: &Expr| {
    matches!(e, Expr::Identifier(s) if s == "Infinity")
      || matches!(e, Expr::FunctionCall { name, args: a }
        if name == "DirectedInfinity"
          && a.len() == 1
          && matches!(&a[0], Expr::Integer(1)))
  };
  // An invalid length spec at position 2 emits nninfseq and stays unevaluated.
  let nninfseq = || {
    crate::emit_message(&format!(
      "IntegerPartitions::nninfseq: Position 2 of {} must be All, Infinity, nmax, {{nmin}}, {{nmin, nmax}} or {{nmin, nmax, dn}}, where nmin is a non-negative integer, nmax is a non-negative integer or Infinity, and dn is a nonzero integer.",
      crate::syntax::expr_to_string(&unevaluated("IntegerPartitions", args))
    ));
    Ok(unevaluated("IntegerPartitions", args))
  };

  // Parse length constraints from the second arg into
  // (min parts, max parts, part-count step).
  let (min_len, max_len, len_step): (u64, u64, u64) = if args.len() >= 2 {
    match &args[1] {
      // IntegerPartitions[n, All] / [n, Infinity] — no bounds (0..n parts, so
      // the empty partition of 0 is included).
      Expr::Identifier(s) if s == "All" => (0, n.max(1), 1),
      e if is_pos_inf(e) => (0, n.max(1), 1),
      // IntegerPartitions[n, k] — at most k parts (0..k, so 0 parts qualify).
      e if expr_to_i128(e).is_some() => match expr_to_i128(e).unwrap() {
        k if k >= 0 => (0, k as u64, 1),
        _ => return nninfseq(),
      },
      // IntegerPartitions[n, {k}] — exactly k parts.
      Expr::List(lst) if lst.len() == 1 => match expr_to_i128(&lst[0]) {
        Some(k) if k >= 0 => (k as u64, k as u64, 1),
        _ => return nninfseq(),
      },
      // IntegerPartitions[n, {kmin, kmax}] — range of parts (kmax may be
      // Infinity, meaning no upper bound).
      Expr::List(lst) if lst.len() == 2 => {
        let hi = if is_pos_inf(&lst[1]) {
          Some(n as i128)
        } else {
          expr_to_i128(&lst[1])
        };
        match (expr_to_i128(&lst[0]), hi) {
          (Some(lo), Some(hi)) if lo >= 0 && hi >= 0 => {
            (lo as u64, hi as u64, 1)
          }
          _ => return nninfseq(),
        }
      }
      // IntegerPartitions[n, {kmin, kmax, dn}] — part counts kmin, kmin+dn, …
      Expr::List(lst) if lst.len() == 3 => {
        let hi = if is_pos_inf(&lst[1]) {
          Some(n as i128)
        } else {
          expr_to_i128(&lst[1])
        };
        match (expr_to_i128(&lst[0]), hi, expr_to_i128(&lst[2])) {
          (Some(lo), Some(hi), Some(dn)) if lo >= 0 && hi >= 0 && dn > 0 => {
            (lo as u64, hi as u64, dn as u64)
          }
          _ => return nninfseq(),
        }
      }
      _ => return nninfseq(),
    }
  } else {
    (1, n.max(1), 1)
  };

  // Parse allowed elements from third arg
  let allowed_signed: Option<Vec<i128>> = if args.len() >= 3 {
    match &args[2] {
      Expr::Identifier(s) if s == "All" => None,
      Expr::List(elems) => {
        let mut vals = Vec::new();
        for e in elems {
          match expr_to_i128(e) {
            Some(v) => vals.push(v),
            _ => {
              return Ok(unevaluated("IntegerPartitions", args));
            }
          }
        }
        vals.sort_unstable();
        vals.dedup();
        vals.reverse(); // descending for generation
        Some(vals)
      }
      _ => {
        return Ok(unevaluated("IntegerPartitions", args));
      }
    }
  } else {
    None
  };

  // Check if allowed set has any non-positive values
  let has_non_positive = allowed_signed
    .as_ref()
    .is_some_and(|v| v.iter().any(|&x| x <= 0));

  // Convert to unsigned if all positive (optimization for common case)
  let allowed: Option<Vec<u64>> = if has_non_positive {
    None // we'll use the signed path
  } else {
    allowed_signed
      .as_ref()
      .map(|v| v.iter().map(|&x| x as u64).collect())
  };

  // Parse max results from fourth arg
  let max_results: Option<usize> = if args.len() >= 4 {
    match expr_to_i128(&args[3]) {
      Some(k) if k >= 0 => Some(k as usize),
      _ => {
        return Ok(unevaluated("IntegerPartitions", args));
      }
    }
  } else {
    None
  };

  // When a part-count step is in effect the count limit must be applied after
  // filtering by step, so let the generators run unbounded in that case.
  let gen_max = if len_step == 1 { max_results } else { None };
  // Keep only partitions whose length matches the step, then truncate.
  let apply_step = |lens: Vec<Vec<u64>>| -> Vec<Vec<u64>> {
    let mut kept: Vec<Vec<u64>> = if len_step == 1 {
      lens
    } else {
      lens
        .into_iter()
        .filter(|p| (p.len() as u64 - min_len).is_multiple_of(len_step))
        .collect()
    };
    if len_step != 1
      && let Some(m) = max_results
    {
      kept.truncate(m);
    }
    kept
  };

  // Special case: n == 0
  // The only partition of 0 is the empty partition {}, which has 0 parts
  if n == 0 {
    if min_len == 0 || (args.len() < 2) {
      return Ok(Expr::List(vec![Expr::List(vec![].into())].into()));
    } else {
      return Ok(Expr::List(vec![].into()));
    }
  }

  if has_non_positive {
    // Use signed partition generator for sets containing non-positive values
    let elems = allowed_signed.as_ref().unwrap();
    let mut result_signed: Vec<Vec<i128>> = Vec::new();
    let mut current_signed: Vec<i128> = Vec::new();
    generate_partitions_restricted_signed(
      n as i128,
      max_len,
      min_len,
      elems,
      0,
      &mut current_signed,
      &mut result_signed,
      gen_max,
    );
    if len_step != 1 {
      result_signed
        .retain(|p| (p.len() as u64 - min_len).is_multiple_of(len_step));
      if let Some(m) = max_results {
        result_signed.truncate(m);
      }
    }
    return Ok(Expr::List(
      result_signed
        .into_iter()
        .map(|p| Expr::List(p.into_iter().map(Expr::Integer).collect()))
        .collect(),
    ));
  }

  let mut result = Vec::new();
  let mut current = Vec::new();

  match &allowed {
    Some(elems) => {
      generate_partitions_restricted(
        n,
        max_len,
        min_len,
        elems,
        0,
        &mut current,
        &mut result,
        gen_max,
      );
    }
    None => {
      generate_partitions(
        n,
        n,
        max_len,
        min_len,
        &mut current,
        &mut result,
        gen_max,
      );
    }
  }

  let result = apply_step(result);
  Ok(Expr::List(
    result
      .into_iter()
      .map(|p| {
        Expr::List(p.into_iter().map(|v| Expr::Integer(v as i128)).collect())
      })
      .collect(),
  ))
}

/// Generate all partitions of `remaining` where each part <= `max_part`,
/// with total number of parts between `min_len` and `max_len`.
fn generate_partitions(
  remaining: u64,
  max_part: u64,
  max_len: u64,
  min_len: u64,
  current: &mut Vec<u64>,
  result: &mut Vec<Vec<u64>>,
  max_results: Option<usize>,
) {
  if max_results.is_some_and(|m| result.len() >= m) {
    return;
  }
  if remaining == 0 {
    if current.len() as u64 >= min_len {
      result.push(current.clone());
    }
    return;
  }
  if current.len() as u64 >= max_len {
    return;
  }
  let upper = remaining.min(max_part);
  for part in (1..=upper).rev() {
    current.push(part);
    generate_partitions(
      remaining - part,
      part,
      max_len,
      min_len,
      current,
      result,
      max_results,
    );
    current.pop();
    if max_results.is_some_and(|m| result.len() >= m) {
      return;
    }
  }
}

/// Generate partitions using only elements from `elems` (sorted descending, positive).
fn generate_partitions_restricted(
  remaining: u64,
  max_len: u64,
  min_len: u64,
  elems: &[u64],
  start_idx: usize,
  current: &mut Vec<u64>,
  result: &mut Vec<Vec<u64>>,
  max_results: Option<usize>,
) {
  if max_results.is_some_and(|m| result.len() >= m) {
    return;
  }
  if remaining == 0 {
    if current.len() as u64 >= min_len {
      result.push(current.clone());
    }
    return;
  }
  if current.len() as u64 >= max_len {
    return;
  }
  // Skip elements that are too large via binary search on the
  // descending-sorted slice (O(log n) vs the previous linear skip).
  let skip = elems[start_idx..].partition_point(|&p| p > remaining);
  let lo = start_idx + skip;
  // Number of additional parts still allowed after we pick one here.
  let parts_left_after = max_len - current.len() as u64 - 1;
  // Fast path: when this is the final part, no need to recurse.
  // A single value equal to `remaining` must exist in `elems[lo..]`
  // (which is sorted descending). Binary-search for it.
  if parts_left_after == 0 {
    if (current.len() as u64 + 1) >= min_len
      && let Ok(_) = elems[lo..].binary_search_by(|p| remaining.cmp(p))
    {
      current.push(remaining);
      result.push(current.clone());
      current.pop();
    }
    return;
  }
  for i in lo..elems.len() {
    let part = elems[i];
    // Lower-bound prune: with `parts_left_after` more parts available
    // (each <= part, since elems is descending and start_idx grows),
    // the maximum reachable sum is `(parts_left_after + 1) * part`.
    // If that's still below `remaining`, no smaller part can reach it
    // either — break out of the loop entirely.
    if part * (parts_left_after + 1) < remaining {
      break;
    }
    current.push(part);
    generate_partitions_restricted(
      remaining - part,
      max_len,
      min_len,
      elems,
      i,
      current,
      result,
      max_results,
    );
    current.pop();
    if max_results.is_some_and(|m| result.len() >= m) {
      return;
    }
  }
}

/// Generate partitions using signed elements (supports negative and zero values).
/// `elems` should be sorted descending.
fn generate_partitions_restricted_signed(
  remaining: i128,
  max_len: u64,
  min_len: u64,
  elems: &[i128],
  start_idx: usize,
  current: &mut Vec<i128>,
  result: &mut Vec<Vec<i128>>,
  max_results: Option<usize>,
) {
  if max_results.is_some_and(|m| result.len() >= m) {
    return;
  }
  if remaining == 0 && current.len() as u64 >= min_len {
    result.push(current.clone());
    return;
  }
  if current.len() as u64 >= max_len {
    return;
  }
  for i in start_idx..elems.len() {
    let part = elems[i];
    // Skip if adding this part makes the remaining impossible
    // (for positive parts, remaining must stay >= 0 eventually;
    //  for non-positive parts, remaining must stay <= n eventually)
    if part > remaining && !elems[i..].iter().any(|&e| e < 0) {
      continue;
    }
    // For non-positive parts, we need remaining to still be achievable
    let new_remaining = remaining - part;
    // Bound: with (max_len - current.len() - 1) parts left,
    // using the largest element can give at most that * max_elem,
    // and using smallest can give that * min_elem
    let parts_left = max_len as i128 - current.len() as i128 - 1;
    if parts_left >= 0 {
      let max_elem = elems[0]; // largest
      let min_elem = *elems.last().unwrap(); // smallest
      let max_achievable = parts_left * max_elem;
      let min_achievable = parts_left * min_elem;
      if new_remaining > max_achievable || new_remaining < min_achievable {
        continue;
      }
    }
    current.push(part);
    generate_partitions_restricted_signed(
      new_remaining,
      max_len,
      min_len,
      elems,
      i,
      current,
      result,
      max_results,
    );
    current.pop();
    if max_results.is_some_and(|m| result.len() >= m) {
      return;
    }
  }
}

/// Prime factorization of `n` as `(prime, exponent)` pairs, using the fast
/// integer factorizer (Pollard's rho for large `n`). Returns `None` if `n`
/// doesn't fit `i128` or can't be parsed. Far faster than O(√n) trial
/// division for highly-composite or large inputs (e.g. the ~10^20 terms of an
/// aliquot sequence): the cost is factorization + d(n), not √n.
fn prime_factorization_u128(n: u128) -> Option<Vec<(u128, u32)>> {
  if n == 1 {
    return Some(Vec::new());
  }
  let ni = i128::try_from(n).ok()?;
  let factored = factor_integer_i128(ni).ok()?;
  let Expr::List(pairs) = &factored else {
    return None;
  };
  let mut out = Vec::with_capacity(pairs.len());
  for pair in pairs.iter() {
    let Expr::List(pv) = pair else { return None };
    if pv.len() != 2 {
      return None;
    }
    let (p, e) = (expr_to_i128(&pv[0])?, expr_to_i128(&pv[1])?);
    if p == 1 {
      continue; // FactorInteger[1] yields {{1, 1}}
    }
    if p < 2 || e < 1 {
      return None;
    }
    out.push((p as u128, e as u32));
  }
  Some(out)
}

/// All divisors of `n` (sorted ascending), via fast factorization. Returns
/// `None` to signal the caller should fall back (only on un-factorable input).
fn divisors_u128(n: u128) -> Option<Vec<u128>> {
  let factors = prime_factorization_u128(n)?;
  let mut divs = vec![1u128];
  for &(p, e) in &factors {
    let base = divs.clone();
    let mut pk = 1u128;
    for _ in 0..e {
      pk = pk.checked_mul(p)?;
      for &d in &base {
        divs.push(d.checked_mul(pk)?);
      }
    }
  }
  divs.sort_unstable();
  Some(divs)
}

/// The value of a `GaussianIntegers -> True/False` option argument.
fn gaussian_integers_option(opt: &Expr) -> Option<bool> {
  let value = |v: &Expr| match v {
    Expr::Identifier(s) if s == "True" => Some(true),
    Expr::Identifier(s) if s == "False" => Some(false),
    _ => None,
  };
  if let Expr::Rule {
    pattern,
    replacement,
  } = opt
    && matches!(pattern.as_ref(), Expr::Identifier(s) if s == "GaussianIntegers")
  {
    return value(replacement);
  }
  if let Expr::FunctionCall { name, args } = opt
    && (name == "Rule" || name == "RuleDelayed")
    && args.len() == 2
    && matches!(&args[0], Expr::Identifier(s) if s == "GaussianIntegers")
  {
    return value(&args[1]);
  }
  None
}

fn divisors_gaussian_expr(
  a: i128,
  b: i128,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let Some(divs) = gaussian_divisors(a, b) else {
    return Ok(unevaluated("Divisors", args));
  };
  let mut items = Vec::with_capacity(divs.len());
  for (re, im) in divs {
    items.push(gaussian_to_expr(re, im)?);
  }
  Ok(Expr::List(items.into()))
}

pub fn divisors_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Divisors[n, GaussianIntegers -> True] lists divisors over ℤ[i];
  // GaussianIntegers -> False is the ordinary form.
  if args.len() == 2 {
    return match gaussian_integers_option(&args[1]) {
      Some(true) => match extract_gaussian_integer(&args[0]) {
        Some((a, b)) if (a, b) != (0, 0) => divisors_gaussian_expr(a, b, args),
        _ => Ok(unevaluated("Divisors", args)),
      },
      Some(false) => divisors_ast(&args[..1]),
      None => Ok(unevaluated("Divisors", args)),
    };
  }
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Divisors expects exactly 1 argument".into(),
    ));
  }

  // Divisors[z] for a proper Gaussian integer lists one first-quadrant
  // representative per associate class.
  if let Some((a, b)) = extract_gaussian_integer(&args[0])
    && b != 0
  {
    return divisors_gaussian_expr(a, b, args);
  }

  let n = match expr_to_i128(&args[0]) {
    // Divisors[0] is undefined; Wolfram leaves the call unevaluated.
    Some(0) => {
      return Ok(unevaluated("Divisors", args));
    }
    Some(n) => n.unsigned_abs(),
    None => {
      return Ok(unevaluated("Divisors", args));
    }
  };

  let divs = match divisors_u128(n) {
    Some(d) => d,
    None => {
      // Fallback: O(√n) trial division.
      let mut divs = Vec::new();
      let sqrt_n = (n as f64).sqrt() as u128;
      for i in 1..=sqrt_n {
        if n % i == 0 {
          divs.push(i);
          if i != n / i {
            divs.push(n / i);
          }
        }
      }
      divs.sort_unstable();
      divs
    }
  };
  Ok(Expr::List(
    divs.into_iter().map(|d| Expr::Integer(d as i128)).collect(),
  ))
}

fn divisor_sigma_gaussian_expr(
  k_expr: &Expr,
  a: i128,
  b: i128,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("DivisorSigma", args));
  let Some(k) = expr_to_i128(k_expr).filter(|&k| k >= 0) else {
    return unevaluated();
  };
  match gaussian_divisor_sigma(k, a, b) {
    Some((re, im)) => gaussian_to_expr(re, im),
    None => unevaluated(),
  }
}

/// DivisorSigma[k, n] - Returns the sum of the k-th powers of divisors of n
pub fn divisor_sigma_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // DivisorSigma[k, n, GaussianIntegers -> True] and Gaussian-integer n
  // use the multiplicative formula over the canonical ℤ[i] primes.
  if args.len() == 3 {
    return match gaussian_integers_option(&args[2]) {
      Some(true) => match extract_gaussian_integer(&args[1]) {
        Some((a, b)) if (a, b) != (0, 0) => {
          divisor_sigma_gaussian_expr(&args[0], a, b, args)
        }
        _ => Ok(unevaluated("DivisorSigma", args)),
      },
      Some(false) => divisor_sigma_ast(&args[..2]),
      None => Ok(unevaluated("DivisorSigma", args)),
    };
  }
  if args.len() == 2
    && let Some((a, b)) = extract_gaussian_integer(&args[1])
    && b != 0
  {
    return divisor_sigma_gaussian_expr(&args[0], a, b, args);
  }
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "DivisorSigma expects exactly 2 arguments".into(),
    ));
  }

  // The order may be negative. Since divisors pair as d <-> n/d,
  //   DivisorSigma[-p, n] = sum_{d|n} d^{-p} = sigma_p(n) / n^p  (a rational).
  // Non-integer orders (symbolic/rational/real) are not handled here.
  let order = match expr_to_i128(&args[0]) {
    Some(k) => k,
    None => {
      return Err(InterpreterError::EvaluationError(
        "DivisorSigma: first argument must be a non-negative integer".into(),
      ));
    }
  };

  let n = match expr_to_i128(&args[1]) {
    Some(0) => {
      // 0 has no (finite) divisor sum; wolframscript leaves it unevaluated
      // rather than raising an error.
      return Ok(unevaluated("DivisorSigma", args));
    }
    Some(n) => n.unsigned_abs(),
    None => {
      return Ok(unevaluated("DivisorSigma", args));
    }
  };

  // Compute sigma_p(n) = sum of p-th powers of divisors, where p = |order|.
  let p = order.unsigned_abs() as u32;
  let mut sum: u128 = 0;
  match divisors_u128(n) {
    Some(divs) => {
      for d in divs {
        sum += d.pow(p);
      }
    }
    None => {
      // Fallback: O(√n) trial division.
      let sqrt_n = (n as f64).sqrt() as u128;
      for i in 1..=sqrt_n {
        if n % i == 0 {
          sum += i.pow(p);
          if i != n / i {
            sum += (n / i).pow(p);
          }
        }
      }
    }
  }

  if order >= 0 {
    Ok(Expr::Integer(sum as i128))
  } else {
    // Negative order: divide by n^p to get the exact rational sigma_p(n)/n^p.
    let denom = n.pow(p);
    Ok(make_rational(sum as i128, denom as i128))
  }
}

/// MoebiusMu[n] - Returns the Möbius function value
pub fn moebius_mu_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "MoebiusMu expects exactly 1 argument".into(),
    ));
  }

  // Integer arguments only: MoebiusMu[0] is 0, negative n uses |n|
  // (matching wolframscript). Anything else stays unevaluated.
  let is_integer = matches!(&args[0], Expr::Integer(_) | Expr::BigInteger(_));
  let n = match expr_to_i128(&args[0]) {
    Some(0) if is_integer => return Ok(Expr::Integer(0)),
    Some(v) if is_integer => v.unsigned_abs(),
    _ => {
      return Ok(unevaluated("MoebiusMu", args));
    }
  };

  if n == 1 {
    return Ok(Expr::Integer(1));
  }

  let mut num = n;
  let mut prime_count = 0;

  // Check for factor 2
  if num % 2 == 0 {
    prime_count += 1;
    num /= 2;
    if num % 2 == 0 {
      return Ok(Expr::Integer(0)); // Has squared factor
    }
  }

  // Check odd factors
  let mut i: u128 = 3;
  while i * i <= num {
    if num % i == 0 {
      prime_count += 1;
      num /= i;
      if num % i == 0 {
        return Ok(Expr::Integer(0)); // Has squared factor
      }
    }
    i += 2;
  }

  if num > 1 {
    prime_count += 1;
  }

  Ok(Expr::Integer(if prime_count % 2 == 0 { 1 } else { -1 }))
}

/// EulerPhi[n] - Returns Euler's totient function
pub fn euler_phi_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "EulerPhi expects exactly 1 argument".into(),
    ));
  }

  if let Some(n) = expr_to_i128(&args[0]) {
    if n == 0 {
      return Ok(Expr::Integer(0));
    }
    let mut num = n.unsigned_abs();
    let mut result = num;
    let mut p: u128 = 2;
    while p * p <= num {
      if num % p == 0 {
        while num % p == 0 {
          num /= p;
        }
        result -= result / p;
      }
      p += 1;
    }
    if num > 1 {
      result -= result / num;
    }
    return Ok(Expr::Integer(result as i128));
  }

  if let Expr::BigInteger(n) = &args[0] {
    return euler_phi_bigint(n).map(bigint_to_expr);
  }

  let call = unevaluated("EulerPhi", args);
  // A concrete non-integer number emits ::int like wolframscript
  // (EulerPhi[1/2], EulerPhi[2.5]); symbolic arguments stay silent.
  if is_concrete_number(&args[0]) {
    crate::emit_message(&format!(
      "EulerPhi::int: Integer expected at position 1 in {}.",
      crate::syntax::expr_to_message_form(&call)
    ));
  }
  Ok(call)
}

/// Compute Euler's totient function for a BigInt by factoring and applying
/// φ(n) = ∏ pᵢ^(eᵢ-1) (pᵢ - 1).
fn euler_phi_bigint(n: &BigInt) -> Result<BigInt, InterpreterError> {
  use num_traits::{One, Zero};
  if n.is_zero() {
    return Ok(BigInt::zero());
  }
  let n_abs = n.abs();
  if n_abs == BigInt::one() {
    return Ok(BigInt::one());
  }
  let factors_expr = factor_integer_bigint(&n_abs)?;
  let factors = match &factors_expr {
    Expr::List(v) => v,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "EulerPhi: unexpected factorization result".into(),
      ));
    }
  };
  let mut result = BigInt::one();
  for f in factors {
    let Expr::List(pair) = f else { continue };
    if pair.len() != 2 {
      continue;
    }
    let Some(p) = expr_to_bigint(&pair[0]) else {
      continue;
    };
    if p == BigInt::from(-1) {
      continue;
    }
    let Some(e) = expr_to_i128(&pair[1]) else {
      continue;
    };
    if e < 1 {
      continue;
    }
    let p_minus_one = &p - BigInt::one();
    let p_pow = if e == 1 {
      BigInt::one()
    } else {
      p.pow((e - 1) as u32)
    };
    result = result * p_pow * p_minus_one;
  }
  Ok(result)
}

/// Compute Euler's totient function for a positive integer.
pub fn euler_phi_i128(n: i128) -> i128 {
  if n <= 0 {
    return 0;
  }
  let mut num = n as u128;
  let mut result = n as u128;
  let mut p: u128 = 2;
  while p * p <= num {
    if num.is_multiple_of(p) {
      while num.is_multiple_of(p) {
        num /= p;
      }
      result -= result / p;
    }
    p += 1;
  }
  if num > 1 {
    result -= result / num;
  }
  result as i128
}

/// CarmichaelLambda[n] - Compute the Carmichael lambda function.
/// The smallest positive integer m such that a^m ≡ 1 (mod n) for all a coprime to n.
pub fn carmichael_lambda_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "CarmichaelLambda expects exactly 1 argument".into(),
    ));
  }

  let n = match expr_to_i128(&args[0]) {
    Some(0) => return Ok(Expr::Integer(0)),
    Some(n) => n.unsigned_abs(),
    _ => {
      return Ok(unevaluated("CarmichaelLambda", args));
    }
  };

  Ok(Expr::Integer(carmichael_lambda_u128(n) as i128))
}

/// Carmichael lambda λ(n): the smallest m with a^m ≡ 1 (mod n) for all a
/// coprime to n. λ(0) is defined here as 0 and λ(1) as 1.
fn carmichael_lambda_u128(n: u128) -> u128 {
  if n == 0 {
    return 0;
  }
  if n <= 1 {
    return 1;
  }

  // Factor n into prime powers, accumulating the LCM of λ(p^k).
  let mut remaining = n;
  let mut result: u128 = 1;

  let mut p: u128 = 2;
  while p * p <= remaining {
    if remaining.is_multiple_of(p) {
      let mut pk: u128 = 1;
      while remaining.is_multiple_of(p) {
        pk *= p;
        remaining /= p;
      }
      let lambda_pk = if p == 2 {
        if pk <= 2 {
          1 // lambda(2) = 1
        } else if pk == 4 {
          2 // lambda(4) = 2
        } else {
          pk / 4 // lambda(2^k) = 2^(k-2) for k >= 3
        }
      } else {
        // lambda(p^k) = phi(p^k) = p^(k-1) * (p - 1)
        (pk / p) * (p - 1)
      };
      result = lcm_u128(result, lambda_pk);
    }
    p += 1;
  }
  if remaining > 1 {
    // remaining is a prime p with exponent 1: lambda(p) = p - 1
    result = lcm_u128(result, remaining - 1);
  }

  result
}

fn lcm_u128(a: u128, b: u128) -> u128 {
  if a == 0 || b == 0 {
    0
  } else {
    a / gcd_u128(a, b) * b
  }
}

/// JacobiSymbol[n, m] - Compute the Jacobi symbol (n/m)
pub fn jacobi_symbol_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiSymbol expects exactly 2 arguments".into(),
    ));
  }

  let n = match expr_to_i128(&args[0]) {
    Some(v) => v,
    None => {
      return Ok(unevaluated("JacobiSymbol", args));
    }
  };

  let m = match expr_to_i128(&args[1]) {
    Some(v) => v,
    None => {
      return Ok(unevaluated("JacobiSymbol", args));
    }
  };

  // wolframscript defines JacobiSymbol[a, n] for all integers n via the
  // Kronecker-symbol extension (even/zero/negative n included); for odd
  // positive n it coincides with the ordinary Jacobi symbol.
  Ok(Expr::Integer(kronecker_symbol(n, m)))
}

/// Compute the Jacobi symbol (a/n) using the standard algorithm
fn jacobi_symbol(mut a: i128, mut n: i128) -> i128 {
  if n == 1 {
    return 1;
  }
  if a == 0 {
    return 0;
  }

  // Reduce a mod n (keep non-negative)
  a = ((a % n) + n) % n;
  let mut result: i128 = 1;

  while a != 0 {
    // Factor out powers of 2
    while a % 2 == 0 {
      a /= 2;
      let n_mod_8 = n % 8;
      if n_mod_8 == 3 || n_mod_8 == 5 {
        result = -result;
      }
    }
    // Apply quadratic reciprocity
    std::mem::swap(&mut a, &mut n);
    if a % 4 == 3 && n % 4 == 3 {
      result = -result;
    }
    a %= n;
  }

  if n == 1 { result } else { 0 }
}

/// Compute the Kronecker symbol (a/n), the extension of the Jacobi symbol to
/// all integers n (including even, zero and negative). wolframscript defines
/// both `JacobiSymbol[a, n]` and `KroneckerSymbol[a, n]` by this same symbol.
pub fn kronecker_symbol(a: i128, n: i128) -> i128 {
  if n == 0 {
    return if a == 1 || a == -1 { 1 } else { 0 };
  }
  if n == 1 {
    return 1;
  }
  if n == -1 {
    return if a < 0 { -1 } else { 1 };
  }

  // (a/2): 0 for even a, +1 for a ≡ ±1 (mod 8), -1 for a ≡ ±3 (mod 8).
  if n == 2 {
    if a % 2 == 0 {
      return 0;
    }
    let a_mod_8 = a.rem_euclid(8);
    return if a_mod_8 == 1 || a_mod_8 == 7 { 1 } else { -1 };
  }
  if n == -2 {
    return kronecker_symbol(a, -1) * kronecker_symbol(a, 2);
  }

  // For negative n, factor out the sign.
  if n < 0 {
    return kronecker_symbol(a, -1) * kronecker_symbol(a, -n);
  }

  // n > 2 and positive. Factor out the powers of 2, then use the Jacobi symbol
  // for the remaining odd part.
  let mut n_rem = n;
  let mut result: i128 = 1;

  let mut twos = 0;
  while n_rem % 2 == 0 {
    n_rem /= 2;
    twos += 1;
  }
  if twos > 0 {
    let k2 = kronecker_symbol(a, 2);
    for _ in 0..twos {
      result *= k2;
    }
  }

  if n_rem > 1 {
    result *= jacobi_symbol(a, n_rem);
  }

  result
}

/// CoprimeQ[a, b, ...] - Tests if integers are pairwise coprime
pub fn coprime_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "CoprimeQ expects at least 1 argument".into(),
    ));
  }

  // Single argument: CoprimeQ[n] tests whether n is a unit (GCD[n] == 1),
  // which for integers means |n| == 1. wolframscript: CoprimeQ[1] = True,
  // CoprimeQ[5] = CoprimeQ[x] = False. This case is reached on its own and
  // as the per-element result of the Listable threading of a single list,
  // e.g. CoprimeQ[{6, 35}] -> {CoprimeQ[6], CoprimeQ[35]} -> {False, False}.
  if args.len() == 1 {
    let is_unit =
      matches!(expr_to_i128(&args[0]), Some(n) if n.unsigned_abs() == 1);
    return Ok(bool_expr(is_unit));
  }

  // Gaussian-integer CoprimeQ: when at least one argument is a complex
  // (Gaussian) integer, two values are coprime iff their gcd over Z[i] is a
  // unit (the canonical associate 1). Coprimality is tested pairwise.
  if let Some(gs) = args
    .iter()
    .map(expr_to_gaussian_int)
    .collect::<Option<Vec<_>>>()
    && gs.iter().any(|(_, im)| *im != 0)
  {
    for i in 0..gs.len() {
      for j in (i + 1)..gs.len() {
        if gaussian_gcd(gs[i], gs[j]) != (1, 0) {
          return Ok(bool_expr(false));
        }
      }
    }
    return Ok(bool_expr(true));
  }

  // Extract all integer values
  let nums: Vec<u128> = args
    .iter()
    .filter_map(|a| expr_to_i128(a).map(|n| n.unsigned_abs()))
    .collect();

  if nums.len() != args.len() {
    return Ok(unevaluated("CoprimeQ", args));
  }

  // Check all pairs are coprime, i.e. GCD == 1. A zero argument must NOT be
  // coerced to 1: GCD[0, n] = |n|, so CoprimeQ[0, 5] is False and only
  // CoprimeQ[0, 1] (GCD 1) is True. The Euclidean loop already handles a
  // zero operand, leaving the other value as the gcd.
  for i in 0..nums.len() {
    for j in (i + 1)..nums.len() {
      let (mut a, mut b) = (nums[i], nums[j]);
      while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
      }
      if a != 1 {
        return Ok(bool_expr(false));
      }
    }
  }

  Ok(bool_expr(true))
}

/// Binomial[n, k] - Binomial coefficient
pub fn binomial_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Binomial expects exactly 2 arguments".into(),
    ));
  }
  // Binomial[k, k] = 1 for any k (including symbolic), matching wolframscript.
  // Inexact arguments (machine reals) yield 1. rather than the exact integer.
  if expr_to_string(&args[0]) == expr_to_string(&args[1]) {
    let inexact = matches!(&args[0], Expr::Real(_) | Expr::BigFloat(_, _))
      || matches!(&args[1], Expr::Real(_) | Expr::BigFloat(_, _));
    return Ok(if inexact {
      Expr::Real(1.0)
    } else {
      Expr::Integer(1)
    });
  }
  match (expr_to_i128(&args[0]), expr_to_i128(&args[1])) {
    (Some(n), Some(k)) => Ok(crate::functions::math_ast::bigint_to_expr(
      binomial_coeff_big(n, k),
    )),
    (None, Some(k)) if k >= 0 => {
      // Generalized binomial for non-integer n with non-negative integer k:
      // Binomial[n, k] = n*(n-1)*...*(n-k+1) / k!
      // Works for Rational n, symbolic n, etc.
      let k = k as usize;
      if k == 0 {
        return Ok(Expr::Integer(1));
      }
      // Compute falling factorial: n * (n-1) * ... * (n-k+1)
      let mut numer_factors: Vec<Expr> = Vec::with_capacity(k);
      for i in 0..k {
        let factor =
          super::plus_ast(&[args[0].clone(), Expr::Integer(-(i as i128))])?;
        numer_factors.push(factor);
      }
      let numer = super::times_ast(&numer_factors)?;
      let mut denom_val: i128 = 1;
      for i in 1..=(k as i128) {
        denom_val *= i;
      }
      let denom = Expr::Integer(denom_val);
      super::divide_ast(&[numer, denom])
    }
    _ => {
      // Try Real evaluation using Gamma function
      let n_f64 = match &args[0] {
        Expr::Real(f) => Some(*f),
        Expr::Integer(n) => Some(*n as f64),
        _ => crate::functions::math_ast::try_eval_to_f64(&args[0]),
      };
      let k_f64 = match &args[1] {
        Expr::Real(f) => Some(*f),
        Expr::Integer(n) => Some(*n as f64),
        _ => crate::functions::math_ast::try_eval_to_f64(&args[1]),
      };
      if let (Some(n), Some(k)) = (n_f64, k_f64) {
        // A negative integer n (as a machine real) with a non-negative integer
        // k collides Gamma poles in both the numerator (Gamma[n+1]) and the
        // denominator (Gamma[n-k+1]), so the Gamma-ratio path yields NaN/Inf.
        // Use the finite falling-factorial definition instead:
        //   Binomial[n, k] = n (n-1) ... (n-k+1) / k!
        if n < 0.0
          && n.fract() == 0.0
          && k >= 0.0
          && k.fract() == 0.0
          && k <= 170.0
        {
          let kk = k as i128;
          let mut numer = 1.0;
          for i in 0..kk {
            numer *= n - i as f64;
          }
          let mut fact = 1.0;
          for i in 1..=kk {
            fact *= i as f64;
          }
          return Ok(Expr::Real(numer / fact));
        }
        // Poles of Gamma in numerator/denominator: a Gamma at a
        // non-positive integer is ComplexInfinity. If the denominator is
        // infinite (and the numerator is not), Binomial is 0.
        let is_nonpos_integer = |x: f64| {
          x <= 0.0 && (x.fract() == 0.0 || (x - x.round()).abs() < 1e-12)
        };
        let num_pole = is_nonpos_integer(n + 1.0);
        let den_pole_k = is_nonpos_integer(k + 1.0);
        let den_pole_nk = is_nonpos_integer(n - k + 1.0);
        if (den_pole_k || den_pole_nk) && !num_pole {
          return Ok(Expr::Integer(0));
        }
        // Only the numerator has a pole → ComplexInfinity.
        if num_pole && !den_pole_k && !den_pole_nk {
          return Ok(Expr::Identifier("ComplexInfinity".to_string()));
        }
        // Binomial[n, k] = Gamma[n+1] / (Gamma[k+1] * Gamma[n-k+1])
        // Use log-gamma for better precision
        fn lgamma(x: f64) -> f64 {
          // Lanczos approximation for log-gamma
          if x < 0.5 {
            let reflection =
              (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln();
            reflection - lgamma(1.0 - x)
          } else {
            let x = x - 1.0;
            let g = 7.0_f64;
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
            0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t
              + sum.ln()
          }
        }
        let log_result =
          lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0);
        let result = log_result.exp();
        Ok(Expr::Real(result))
      } else {
        Ok(unevaluated("Binomial", args))
      }
    }
  }
}

/// Compute a binomial coefficient as an arbitrary-precision BigInt, so large
/// results (e.g. Binomial[1000, 500], 300 digits) don't overflow i128. The
/// generalized negative-argument identities match `binomial_coeff`.
pub fn binomial_coeff_big(n: i128, k: i128) -> BigInt {
  if k < 0 {
    if n >= 0 {
      return BigInt::from(0);
    }
    let negate = (n + k) % 2 != 0;
    let r = binomial_coeff_big(-k - 1, -n - 1);
    return if negate { -r } else { r };
  }
  if k == 0 {
    return BigInt::from(1);
  }
  if n >= 0 {
    if k > n {
      return BigInt::from(0);
    }
    // Use the smaller of k and n-k for efficiency.
    let k = k.min(n - k);
    let mut result = BigInt::from(1);
    for i in 0..k {
      result = result * BigInt::from(n - i) / BigInt::from(i + 1);
    }
    result
  } else {
    // Generalized: Binomial[-n, k] = (-1)^k * Binomial[n+k-1, k].
    let negate = k % 2 != 0;
    let r = binomial_coeff_big(-n + k - 1, k);
    if negate { -r } else { r }
  }
}

/// Compute binomial coefficient for arbitrary integers (generalized)
pub fn binomial_coeff(n: i128, k: i128) -> i128 {
  if k < 0 {
    // Binomial[n, k] with negative k is zero unless n is also a negative
    // integer, where wolframscript uses
    //   Binomial[n, k] = (-1)^(n+k) Binomial[-k-1, -n-1].
    // e.g. Binomial[-3, -3] = 1 and Binomial[-3, -5] = 6.
    if n >= 0 {
      return 0;
    }
    let sign = if (n + k) % 2 == 0 { 1 } else { -1 };
    return sign * binomial_coeff(-k - 1, -n - 1);
  }
  if k == 0 {
    return 1;
  }
  if n >= 0 {
    if k > n {
      return 0;
    }
    // Use the smaller of k and n-k for efficiency
    let k = k.min(n - k);
    let mut result: i128 = 1;
    for i in 0..k {
      result = result * (n - i) / (i + 1);
    }
    result
  } else {
    // Generalized: Binomial[-n, k] = (-1)^k * Binomial[n+k-1, k]
    let sign = if k % 2 == 0 { 1 } else { -1 };
    sign * binomial_coeff(-n + k - 1, k)
  }
}

// ─── Multinomial ───────────────────────────────────────────────────

/// Multinomial[n1, n2, ...] - Multinomial coefficient (n1+n2+...)! / (n1! * n2! * ...)
///
/// Multinomial has the `Orderless` attribute in Wolfram, so arguments are
/// canonically sorted before evaluation. Numbers sort before symbols, and
/// symbols sort lexicographically.
pub fn multinomial_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() == 1 {
    return Ok(Expr::Integer(1));
  }

  // Orderless: sort args canonically (numbers first, then symbolic by sort key).
  let mut sorted_args = args.to_vec();
  sorted_args
    .sort_by(crate::functions::list_helpers_ast::sorting::canonical_cmp);
  let args = sorted_args.as_slice();

  let unevaluated = || unevaluated("Multinomial", args);

  // Fast path: all integers → integer arithmetic via the cumulative
  // binomial product Multinomial[n1,..,nk] = prod_j Binomial[n1+..+nj, nj].
  // This is valid for negative integers too (binomial_coeff handles the
  // negative cases), e.g. Multinomial[1, 2, -1] = 0 and
  // Multinomial[-3, 1] = -2, matching wolframscript.
  let mut int_vals: Option<Vec<i128>> = Some(Vec::with_capacity(args.len()));
  for a in args {
    if let Expr::Integer(n) = a {
      int_vals.as_mut().unwrap().push(*n);
    } else {
      int_vals = None;
      break;
    }
  }
  if let Some(ints) = int_vals {
    // Accumulate in BigInt so large multinomials don't overflow i128.
    let mut total: i128 = 0;
    let mut result = BigInt::from(1);
    for &n in &ints {
      total += n;
      result *= binomial_coeff_big(total, n);
    }
    return Ok(crate::functions::math_ast::bigint_to_expr(result));
  }

  use crate::functions::math_ast::expr_to_rational;
  let symbolic_indices: Vec<usize> = (0..args.len())
    .filter(|&i| expr_to_rational(&args[i]).is_none())
    .collect();

  // 2-arg symbolic case keeps a single Binomial form for backward
  // compatibility with the existing convention.
  if args.len() == 2 && symbolic_indices.len() >= 2 {
    let sum = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![args[0].clone(), args[1].clone()].into(),
    };
    let binom = Expr::FunctionCall {
      name: "Binomial".to_string(),
      args: vec![sum, args[1].clone()].into(),
    };
    return crate::evaluator::evaluate_expr_to_expr(&binom);
  }

  if symbolic_indices.len() >= 2 {
    return Ok(unevaluated());
  }

  // From here on: 0 or 1 symbolic, the rest are integers/rationals.
  if symbolic_indices.len() == 1 {
    let sym_idx = symbolic_indices[0];
    let sym = args[sym_idx].clone();
    let rest: Vec<Expr> = args
      .iter()
      .enumerate()
      .filter(|(i, _)| *i != sym_idx)
      .map(|(_, e)| e.clone())
      .collect();

    // Compute Multinomial(rest) — must reduce to a number for us to
    // factor cleanly.
    let multi_rest = multinomial_ast(&rest)?;
    if expr_to_rational(&multi_rest).is_none() {
      return Ok(unevaluated());
    }

    // Build sum_total = sym + sum(rest) and the leading Binomial.
    let mut sum_args = Vec::with_capacity(args.len());
    sum_args.push(sym.clone());
    sum_args.extend(rest.iter().cloned());
    let sum_total =
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Plus".to_string(),
        args: sum_args.into(),
      })?;
    let binom = Expr::FunctionCall {
      name: "Binomial".to_string(),
      args: vec![sum_total, sym].into(),
    };
    let product = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![binom, multi_rest].into(),
    };
    return crate::evaluator::evaluate_expr_to_expr(&product);
  }

  // All args are integers/rationals (mixed). Iterate Multinomial as a
  // product of Binomials, preferring integer args in the k position so
  // each Binomial reduces.
  let mut sorted = args.to_vec();
  sorted.sort_by_key(|a| if matches!(a, Expr::Integer(_)) { 0 } else { 1 });
  let mut result: Expr = Expr::Integer(1);
  let mut remaining: Vec<Expr> = sorted;
  while remaining.len() > 1 {
    let first = remaining[0].clone();
    let sum_expr =
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Plus".to_string(),
        args: remaining.clone().into(),
      })?;
    let binom_expr =
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Binomial".to_string(),
        args: vec![sum_expr, first].into(),
      })?;
    result = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![result, binom_expr].into(),
    })?;
    remaining.remove(0);
  }
  Ok(result)
}

// ─── PowerMod ──────────────────────────────────────────────────────

/// PowerMod[a, b, m] - Modular exponentiation: a^b mod m
pub fn power_mod_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "PowerMod expects exactly 3 arguments".into(),
    ));
  }
  // Modular root: PowerMod[a, 1/n, m] returns an x with x^n ≡ a (mod m), or an
  // unevaluated result with a PowerMod::root message when no such x exists.
  if let Some(n) = rational_reciprocal_exponent(&args[1]) {
    return power_mod_root_ast(args, n);
  }
  match (
    expr_to_bigint(&args[0]),
    expr_to_bigint(&args[1]),
    expr_to_bigint(&args[2]),
  ) {
    (Some(base), Some(exp), Some(modulus)) => {
      use num_traits::Zero;
      if modulus.is_zero() {
        crate::emit_message(&format!(
          "PowerMod::divz: The argument 0 in PowerMod[{}, {}, 0] should be nonzero.",
          expr_to_string(&args[0]),
          expr_to_string(&args[1])
        ));
        return Ok(unevaluated("PowerMod", args));
      }
      if exp < BigInt::from(0) {
        // Negative exponent: need modular inverse
        // Try i128 path for mod_inverse
        use num_traits::ToPrimitive;
        match (base.to_i128(), modulus.to_i128()) {
          (Some(b), Some(m)) => {
            if let Some(inv) = mod_inverse(b, m) {
              let pos_exp = (-exp).to_u128().unwrap_or(0);
              let result =
                mod_pow_unsigned(inv as u128, pos_exp, m.unsigned_abs());
              Ok(Expr::Integer(result as i128))
            } else {
              crate::emit_message(&format!(
                "PowerMod::ninv: {} is not invertible modulo {}.",
                expr_to_string(&args[0]),
                expr_to_string(&args[2])
              ));
              Ok(unevaluated("PowerMod", args))
            }
          }
          _ => Ok(unevaluated("PowerMod", args)),
        }
      } else {
        let result = base.modpow(&exp, &modulus);
        // Ensure non-negative result
        let result = ((result % &modulus) + &modulus) % &modulus;
        Ok(bigint_to_expr(result))
      }
    }
    _ => Ok(unevaluated("PowerMod", args)),
  }
}

/// If `expr` is a unit fraction 1/n (stored as Rational[1, n]) with n ≥ 2,
/// return n; otherwise None.
fn rational_reciprocal_exponent(expr: &Expr) -> Option<u128> {
  use num_traits::ToPrimitive;
  if let Expr::FunctionCall { name, args } = expr
    && name == "Rational"
    && args.len() == 2
    && expr_to_bigint(&args[0]) == Some(BigInt::from(1))
  {
    let n = expr_to_bigint(&args[1])?.to_u128()?;
    if n >= 2 {
      return Some(n);
    }
  }
  None
}

/// Largest modulus for which the modular root is found by exhaustive search.
/// Beyond this the call is left unevaluated to avoid long scans.
const MOD_SQRT_BRUTE_LIMIT: u128 = 100_000_000;

/// PowerMod[a, 1/n, m] - a modular n-th root of a modulo m.
///
/// wolframscript returns the unique root via x = a^(n^-1 mod λ(m)) when a is a
/// unit and gcd(n, λ(m)) = 1, and emits a `PowerMod::root` message when no root
/// exists. When several roots exist and gcd(n, λ(m)) ≠ 1 (n ≥ 3),
/// wolframscript's specific choice depends on per-prime CRT lifting and is not
/// reproducible here, so the call is left unevaluated rather than returning a
/// possibly-wrong root. The n = 2 case returns the smallest root, matching
/// wolframscript.
fn power_mod_root_ast(
  args: &[Expr],
  n: u128,
) -> Result<Expr, InterpreterError> {
  use num_traits::{ToPrimitive, Zero};
  let unevaluated = || Ok(unevaluated("PowerMod", args));
  let exp_str = || expr_to_string(&args[1]);
  let (Some(a), Some(m)) = (expr_to_bigint(&args[0]), expr_to_bigint(&args[2]))
  else {
    return unevaluated();
  };
  if m.is_zero() {
    crate::emit_message(&format!(
      "PowerMod::divz: The argument 0 in PowerMod[{}, {}, 0] should be nonzero.",
      expr_to_string(&args[0]),
      exp_str()
    ));
    return unevaluated();
  }
  let m_abs = m.magnitude().clone();
  let Some(m_u) = m_abs.to_u128() else {
    return unevaluated();
  };
  if m_u > MOD_SQRT_BRUTE_LIMIT {
    return unevaluated();
  }
  // Reduce a into [0, m).
  let a_red = ((&a % &m) + &m) % &m;
  let a_u = a_red.to_u128().unwrap_or(0);

  // Closed form for a unit with gcd(n, λ(m)) = 1: a^(n^-1 mod λ(m)) mod m is
  // the unique root and matches wolframscript exactly.
  if gcd_u128(a_u, m_u) == 1 {
    let lambda = carmichael_lambda_u128(m_u);
    if lambda > 0
      && let Some(n_inv) = mod_inverse((n % lambda) as i128, lambda as i128)
    {
      let b = mod_pow_unsigned(a_u, n_inv as u128, m_u);
      if mod_pow_unsigned(b, n, m_u) == a_u {
        return Ok(Expr::Integer(b as i128));
      }
    }
  }

  // Otherwise search for the smallest root. For n == 2 this matches
  // wolframscript; for n ≥ 3 with multiple roots the chosen root may differ,
  // so only the n == 2 result is returned.
  let mut smallest: Option<u128> = None;
  for x in 0..m_u {
    if mod_pow_unsigned(x, n, m_u) == a_u {
      smallest = Some(x);
      break;
    }
  }
  match smallest {
    Some(x) if n == 2 => Ok(Expr::Integer(x as i128)),
    Some(_) => unevaluated(),
    None => {
      crate::emit_message(&format!(
        "PowerMod::root: The equation x^{} = {} (mod {}) has no integer solutions.",
        n,
        expr_to_string(&args[0]),
        expr_to_string(&args[2])
      ));
      unevaluated()
    }
  }
}

/// Binary exponentiation: base^exp mod modulus (all unsigned)
/// Uses BigUint for intermediate multiplication to avoid u128 overflow.
fn mod_pow_unsigned(base: u128, mut exp: u128, modulus: u128) -> u128 {
  use num_bigint::BigUint;
  use num_traits::ToPrimitive;

  if modulus == 1 {
    return 0;
  }
  let m = BigUint::from(modulus);
  let mut result = BigUint::from(1u32);
  let mut b = BigUint::from(base % modulus);
  while exp > 0 {
    if exp % 2 == 1 {
      result = result * &b % &m;
    }
    exp >>= 1;
    b = &b * &b % &m;
  }
  result.to_u128().unwrap_or(0)
}

/// Extended Euclidean algorithm for modular inverse
fn mod_inverse(a: i128, m: i128) -> Option<i128> {
  let m_abs = m.abs();
  let a = ((a % m_abs) + m_abs) % m_abs;
  let (mut old_r, mut r) = (a, m_abs);
  let (mut old_s, mut s) = (1i128, 0i128);
  while r != 0 {
    let q = old_r / r;
    let temp_r = r;
    r = old_r - q * r;
    old_r = temp_r;
    let temp_s = s;
    s = old_s - q * s;
    old_s = temp_s;
  }
  if old_r != 1 {
    None // No inverse exists
  } else {
    Some(((old_s % m_abs) + m_abs) % m_abs)
  }
}

// ─── MersennePrimeExponent ─────────────────────────────────────────

/// Known Mersenne prime exponents (OEIS A000043): the primes p for which
/// 2^p - 1 is prime.
const MERSENNE_EXPONENTS: &[i128] = &[
  2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281,
  3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 86243,
  110503, 132049, 216091, 756839, 859433, 1257787, 1398269, 2976221, 3021377,
  6972593, 13466917, 20996011, 24036583, 25964951, 30402457, 32582657,
  37156667, 42643801, 43112609, 57885161, 74207281, 77232917, 82589933,
];

/// MersennePrimeExponent[n] - gives the nth Mersenne prime exponent
/// A Mersenne prime is a prime of the form 2^p - 1; this returns p.
pub fn mersenne_prime_exponent_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "MersennePrimeExponent expects exactly 1 argument".into(),
    ));
  }

  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 1 && n <= MERSENNE_EXPONENTS.len() as i128 => n as usize,
    Some(n) if n < 1 => {
      return Err(InterpreterError::EvaluationError(format!(
        "The argument {} should be a positive integer.",
        n
      )));
    }
    _ => {
      return Ok(unevaluated("MersennePrimeExponent", args));
    }
  };

  Ok(Expr::Integer(MERSENNE_EXPONENTS[n - 1]))
}

/// MersennePrimeExponentQ[n] - True if n is a Mersenne prime exponent, i.e.
/// 2^n - 1 is prime. Returns False for anything that is not such an integer
/// (composites, non-exponent primes, non-integers/symbols), matching
/// wolframscript's curated list of known exponents.
pub fn mersenne_prime_exponent_q_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "MersennePrimeExponentQ expects exactly 1 argument".into(),
    ));
  }
  let is_exponent = match expr_to_i128(&args[0]) {
    Some(n) => MERSENNE_EXPONENTS.contains(&n),
    None => false,
  };
  Ok(bool_expr(is_exponent))
}

// ─── PrimePi ───────────────────────────────────────────────────────

/// PrimePi[n] - Counts the number of primes <= n
pub fn prime_pi_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PrimePi expects exactly 1 argument".into(),
    ));
  }
  match expr_to_i128(&args[0]) {
    Some(n) => {
      if n < 2 {
        return Ok(Expr::Integer(0));
      }
      let mut count: i128 = 0;
      for i in 2..=n {
        if is_prime_i128(i) {
          count += 1;
        }
      }
      Ok(Expr::Integer(count))
    }
    None if matches!(&args[0], Expr::Real(_)) => {
      let f = if let Expr::Real(f) = &args[0] {
        *f
      } else {
        unreachable!()
      };
      if f < 2.0 {
        return Ok(Expr::Integer(0));
      }
      let n = f.floor() as i128;
      let mut count: i128 = 0;
      for i in 2..=n {
        if is_prime_i128(i) {
          count += 1;
        }
      }
      Ok(Expr::Integer(count))
    }
    _ => {
      // Try to evaluate symbolic constants (Pi, E, etc.) to f64
      if let Some(f) = crate::functions::math_ast::try_eval_to_f64(&args[0]) {
        if f < 2.0 {
          return Ok(Expr::Integer(0));
        }
        let n = f.floor() as i128;
        let mut count: i128 = 0;
        for i in 2..=n {
          if is_prime_i128(i) {
            count += 1;
          }
        }
        Ok(Expr::Integer(count))
      } else {
        Ok(unevaluated("PrimePi", args))
      }
    }
  }
}

// ─── BigInt Primality (Miller-Rabin) ──────────────────────────────

/// Miller-Rabin primality test for BigInt values.
/// Uses deterministic witnesses for small numbers and a set of strong
/// witnesses that provides correct results for all numbers < 3.317e24,
/// plus additional witnesses for larger numbers.
pub fn is_prime_bigint(n: &BigInt) -> bool {
  use num_traits::{One, Zero};

  let one = BigInt::one();
  let two = &one + &one;
  let three = &two + &one;

  if *n <= one {
    return false;
  }
  if *n == two || *n == three {
    return true;
  }
  if (n % &two).is_zero() || (n % &three).is_zero() {
    return false;
  }

  // Small primes trial division
  let small_primes: &[u64] = &[
    5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
    79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157,
    163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239,
    241, 251,
  ];
  for &p in small_primes {
    let bp = BigInt::from(p);
    if *n == bp {
      return true;
    }
    if (n % &bp).is_zero() {
      return false;
    }
  }

  // Write n-1 = d * 2^r
  let n_minus_1 = n - &one;
  let mut d = n_minus_1.clone();
  let mut r: u64 = 0;
  while (&d % &two).is_zero() {
    d /= &two;
    r += 1;
  }

  // Witness bases — deterministic for numbers < 3.317e24,
  // and probabilistically correct (error < 2^-128) for larger numbers
  let witnesses: &[u64] = &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

  'witness: for &a in witnesses {
    let a_big = BigInt::from(a);
    if a_big >= *n {
      continue;
    }
    let mut x = a_big.modpow(&d, n);
    if x == one || x == n_minus_1 {
      continue;
    }
    for _ in 0..r - 1 {
      x = x.modpow(&two, n);
      if x == n_minus_1 {
        continue 'witness;
      }
    }
    return false;
  }
  true
}

// ─── NextPrime ─────────────────────────────────────────────────────

/// NextPrime[n] - Returns the smallest prime greater than n
/// Negative primes (-2, -3, -5, ...) are included in the search space.
pub fn next_prime_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "NextPrime expects 1 or 2 arguments".into(),
    ));
  }

  let k: i128 = if args.len() == 2 {
    match &args[1] {
      Expr::Integer(k) => *k,
      _ => {
        return Ok(unevaluated("NextPrime", args));
      }
    }
  } else {
    1
  };

  // Handle BigInteger separately (only supports k=1 / forward)
  if let Expr::BigInteger(n) = &args[0] {
    if k == 1 {
      return Ok(bigint_to_expr(next_prime_after_bigint(n)));
    } else if k > 1 {
      let mut current = next_prime_after_bigint(n);
      for _ in 1..k {
        current = next_prime_after_bigint(&current);
      }
      return Ok(bigint_to_expr(current));
    }
    return Ok(unevaluated("NextPrime", args));
  }

  let n = match &args[0] {
    Expr::Integer(n) => *n,
    Expr::Real(f) => f.floor() as i128,
    _ => {
      return Ok(unevaluated("NextPrime", args));
    }
  };

  if k > 0 {
    let mut current = n;
    for _ in 0..k {
      current = next_prime_after(current);
    }
    Ok(Expr::Integer(current))
  } else if k < 0 {
    let mut current = n;
    for _ in 0..(-k) {
      current = prev_prime_before(current);
    }
    Ok(Expr::Integer(current))
  } else {
    // k == 0: return unevaluated
    Ok(unevaluated("NextPrime", args))
  }
}

/// Find the smallest prime > n (including negative primes like -2, -3, -5, ...).
fn next_prime_after(n: i128) -> i128 {
  // For n >= 1: search upward from n+1
  if n >= 1 {
    let mut candidate = n + 1;
    while !is_prime_i128(candidate) {
      candidate += 1;
    }
    return candidate;
  }
  // For n < -2: check negative primes between n and -2 (exclusive of n, inclusive of -2)
  // A negative prime is -p where p is a positive prime.
  // Search from n+1 upward: for each candidate c, check if |c| is prime.
  if n < -2 {
    for c in (n + 1)..=-2 {
      if c < 0 && is_prime_i128(-c) {
        return c;
      }
    }
  }
  // No negative prime found > n, or n is -2, -1, or 0: smallest positive prime is 2
  2
}

/// Find the largest prime < n (including negative primes).
/// The prime sequence is: ..., -7, -5, -3, -2, 2, 3, 5, 7, 11, ...
fn prev_prime_before(n: i128) -> i128 {
  // For n > 2: search downward among positive primes
  if n > 2 {
    let mut candidate = n - 1;
    while candidate >= 2 {
      if is_prime_i128(candidate) {
        return candidate;
      }
      candidate -= 1;
    }
    // No positive prime found < n, so previous is -2
    return -2;
  }
  // For n == 2 or n == 1 or n == 0 or n == -1: previous prime is -2
  if n > -2 {
    return -2;
  }
  // For n == -2: previous prime is -3
  // For n < -2: search downward among negative primes
  let mut candidate = n - 1;
  loop {
    if is_prime_i128(-candidate) {
      return candidate;
    }
    candidate -= 1;
  }
}

/// Find the smallest prime > n for BigInt values.
fn next_prime_after_bigint(n: &BigInt) -> BigInt {
  use num_traits::{One, Zero};

  let one = BigInt::one();
  let two = &one + &one;

  // For positive n, search upward
  if *n >= one {
    let mut candidate = n + &one;
    // Ensure candidate is odd
    if (&candidate % &two).is_zero() {
      candidate += &one;
    }
    // If candidate is 2, check it
    if candidate == two {
      return two;
    }
    loop {
      if is_prime_bigint(&candidate) {
        return candidate;
      }
      candidate += &two;
    }
  }

  // For negative or zero n, delegate to i128 path for small values
  // since BigInt negative values that reach here would be small enough
  let zero = BigInt::zero();
  if *n <= zero {
    return BigInt::from(2);
  }

  unreachable!()
}

// ─── ModularInverse ────────────────────────────────────────────────

/// ModularInverse[a, m] - the modular inverse of a modulo m.
/// Returns k such that a*k ≡ 1 (mod m), or returns unevaluated if no inverse exists.
pub fn modular_inverse_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ModularInverse expects exactly 2 arguments".into(),
    ));
  }
  let unevaluated = || Ok(unevaluated("ModularInverse", args));

  use num_traits::{One, Signed, Zero};

  // Both arguments must be ordinary integers.
  let (a, m) = match (expr_to_bigint(&args[0]), expr_to_bigint(&args[1])) {
    (Some(a), Some(m)) => (a, m),
    _ => {
      crate::emit_message(
        "ModularInverse::minv: The two arguments to ModularInverse must be \
         ordinary or Gaussian integers.",
      );
      return unevaluated();
    }
  };

  // The modulus must be nonzero.
  if m.is_zero() {
    crate::emit_message(&format!(
      "ModularInverse::intnz: Nonzero integer expected at position 2 in \
       ModularInverse[{}, {}].",
      expr_to_string(&args[0]),
      expr_to_string(&args[1])
    ));
    return unevaluated();
  }

  let m_abs = m.abs();

  // The inverse modulo (+/-)1 is 0.
  if m_abs.is_one() {
    return Ok(Expr::Integer(0));
  }

  // Extended Euclidean algorithm. `a` and the modulus must be coprime.
  let (gcd, x, _) = extended_gcd(&a, &m_abs);
  if !gcd.abs().is_one() {
    crate::emit_message(&format!(
      "ModularInverse::ninv: {} is not invertible modulo {}.",
      expr_to_string(&args[0]),
      expr_to_string(&args[1])
    ));
    return unevaluated();
  }

  // Canonical inverse in [0, |m|).
  let mut result = ((x % &m_abs) + &m_abs) % &m_abs;
  // A negative modulus uses the representative in (m, 0].
  if m.is_negative() && !result.is_zero() {
    result -= &m_abs;
  }
  Ok(bigint_to_expr(result))
}

/// Extended GCD: returns (gcd, x, y) such that a*x + b*y = gcd
fn extended_gcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
  use num_traits::Zero;
  if a.is_zero() {
    return (b.clone(), BigInt::zero(), BigInt::from(1));
  }
  let (g, x, y) = extended_gcd(&(b % a), a);
  (g, y - (b / a) * &x, x)
}

// ─── BitLength ─────────────────────────────────────────────────────

/// BitLength[n] - Number of bits in the binary representation
/// For n >= 0: Floor[Log2[n]] + 1, with BitLength[0] = 0
/// For n < 0: BitLength[-n - 1] (2's complement convention)
pub fn bit_length_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "BitLength expects exactly 1 argument".into(),
    ));
  }
  match expr_to_bigint(&args[0]) {
    Some(n) => {
      use num_traits::Zero;
      let val = if n < BigInt::from(0) {
        -n - BigInt::from(1)
      } else {
        n
      };
      if val.is_zero() {
        Ok(Expr::Integer(0))
      } else {
        Ok(Expr::Integer(val.bits() as i128))
      }
    }
    None => Ok(unevaluated("BitLength", args)),
  }
}

// ─── Bitwise operations ──────────────────────────────────────────

/// BitAnd[n1, n2, ...] - bitwise AND
pub fn bit_and_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // OneIdentity: a single argument returns itself (numeric or symbolic).
  if args.len() == 1 {
    return Ok(args[0].clone());
  }
  let mut result: Option<BigInt> = None;
  for arg in args {
    match expr_to_bigint(arg) {
      Some(n) => {
        result = Some(match result {
          Some(r) => r & &n,
          None => n,
        });
      }
      None => {
        return Ok(unevaluated("BitAnd", args));
      }
    }
  }
  match result {
    Some(n) => Ok(bigint_to_expr(n)),
    // BitAnd[] is the identity for AND: -1 (all bits set).
    None => Ok(Expr::Integer(-1)),
  }
}

/// BitOr[n1, n2, ...] - bitwise OR
pub fn bit_or_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // OneIdentity: a single argument returns itself (numeric or symbolic).
  if args.len() == 1 {
    return Ok(args[0].clone());
  }
  let mut result: Option<BigInt> = None;
  for arg in args {
    match expr_to_bigint(arg) {
      Some(n) => {
        result = Some(match result {
          Some(r) => r | &n,
          None => n,
        });
      }
      None => {
        return Ok(unevaluated("BitOr", args));
      }
    }
  }
  match result {
    Some(n) => Ok(bigint_to_expr(n)),
    // BitOr[] is the identity for OR: 0.
    None => Ok(Expr::Integer(0)),
  }
}

/// BitXor[n1, n2, ...] - bitwise XOR
pub fn bit_xor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // OneIdentity: a single argument returns itself (numeric or symbolic).
  if args.len() == 1 {
    return Ok(args[0].clone());
  }
  let mut result: Option<BigInt> = None;
  for arg in args {
    match expr_to_bigint(arg) {
      Some(n) => {
        result = Some(match result {
          Some(r) => r ^ &n,
          None => n,
        });
      }
      None => {
        return Ok(unevaluated("BitXor", args));
      }
    }
  }
  match result {
    Some(n) => Ok(bigint_to_expr(n)),
    // BitXor[] is the identity for XOR: 0.
    None => Ok(Expr::Integer(0)),
  }
}

/// BitNot[n] - bitwise NOT (complement)
pub fn bit_not_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "BitNot expects exactly 1 argument".into(),
    ));
  }
  match expr_to_bigint(&args[0]) {
    Some(n) => Ok(bigint_to_expr(!n)),
    None => Ok(unevaluated("BitNot", args)),
  }
}

/// BitShiftRight[n, k] - shift n right by k bits (default k=1)
pub fn bit_shift_right_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (n_expr, k) = match args.len() {
    1 => (&args[0], 1i64),
    2 => {
      let k_val = match expr_to_bigint(&args[1]) {
        Some(b) => match b.try_into() {
          Ok(v) => v,
          Err(_) => {
            return Ok(unevaluated("BitShiftRight", args));
          }
        },
        None => {
          return Ok(unevaluated("BitShiftRight", args));
        }
      };
      (&args[0], k_val)
    }
    _ => {
      return Ok(unevaluated("BitShiftRight", args));
    }
  };
  match expr_to_bigint(n_expr) {
    Some(n) => {
      if k >= 0 {
        Ok(bigint_to_expr(truncating_shift_right(n, k as usize)))
      } else {
        Ok(bigint_to_expr(n << (-k) as usize))
      }
    }
    None => Ok(unevaluated("BitShiftRight", args)),
  }
}

/// Wolfram's right shift truncates toward zero (BitShiftRight[-7, 1] is
/// -3, BitShiftRight[-1, 1] is 0), unlike the arithmetic `>>` operator
/// which floors (-7 >> 1 == -4). Division by 2^k gives the truncating
/// behavior for either sign.
fn truncating_shift_right(n: BigInt, k: usize) -> BigInt {
  use num_traits::One;
  n / (BigInt::one() << k)
}

/// BitShiftLeft[n, k] - shift n left by k bits (default k=1)
pub fn bit_shift_left_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (n_expr, k) = match args.len() {
    1 => (&args[0], 1i64),
    2 => {
      let k_val = match expr_to_bigint(&args[1]) {
        Some(b) => match b.try_into() {
          Ok(v) => v,
          Err(_) => {
            return Ok(unevaluated("BitShiftLeft", args));
          }
        },
        None => {
          return Ok(unevaluated("BitShiftLeft", args));
        }
      };
      (&args[0], k_val)
    }
    _ => {
      return Ok(unevaluated("BitShiftLeft", args));
    }
  };
  match expr_to_bigint(n_expr) {
    Some(n) => {
      if k >= 0 {
        Ok(bigint_to_expr(n << k as usize))
      } else {
        // Negative shifts are truncating right shifts (see
        // truncating_shift_right): BitShiftLeft[-7, -1] is -3.
        Ok(bigint_to_expr(truncating_shift_right(n, (-k) as usize)))
      }
    }
    None => Ok(unevaluated("BitShiftLeft", args)),
  }
}

// ─── IntegerExponent ──────────────────────────────────────────────

/// FrobeniusNumber[{a1, a2, ...}] - Largest integer that cannot be represented
/// as a non-negative integer linear combination of the given positive integers.
/// Returns Infinity if the GCD is not 1, -1 if 1 is in the set.
pub fn frobenius_number_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(unevaluated("FrobeniusNumber", args));
  }

  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(unevaluated("FrobeniusNumber", args));
    }
  };

  if items.is_empty() {
    crate::emit_message(&format!(
      "FrobeniusNumber::coef: The first argument {} of FrobeniusNumber should be a nonempty list of positive integers.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated("FrobeniusNumber", args));
  }

  // Extract positive integers
  let mut nums: Vec<i128> = Vec::new();
  for item in items {
    match item {
      e if expr_to_i128(e).is_some_and(|n| n > 0) => {
        nums.push(expr_to_i128(e).unwrap())
      }
      _ => {
        crate::emit_message(&format!(
          "FrobeniusNumber::coef: The first argument {} of FrobeniusNumber should be a nonempty list of positive integers.",
          expr_to_string(&args[0])
        ));
        return Ok(unevaluated("FrobeniusNumber", args));
      }
    }
  }

  // If 1 is in the set, every non-negative integer is representable
  if nums.contains(&1) {
    return Ok(Expr::Integer(-1));
  }

  // Compute GCD of all elements

  let mut g = nums[0];
  for &n in &nums[1..] {
    g = gcd(g, n);
  }

  // If GCD > 1, infinitely many integers can't be represented
  if g > 1 {
    return Ok(Expr::Identifier("Infinity".to_string()));
  }

  // For two coprime numbers, use the closed formula: a*b - a - b
  if nums.len() == 2 {
    let (a, b) = (nums[0], nums[1]);
    return Ok(Expr::Integer(a * b - a - b));
  }

  // General case: dynamic programming
  // Upper bound for Frobenius number: a1*a2 - a1 - a2 (using two smallest)
  nums.sort();
  let a0 = nums[0] as usize;

  // Use the round-robin algorithm (Wilf) based on shortest paths
  // Build shortest-path array: n[i] = smallest number representable that is ≡ i (mod a0)
  let mut n_arr = vec![i128::MAX; a0];
  n_arr[0] = 0;

  // BFS/relaxation
  let mut changed = true;
  while changed {
    changed = false;
    for residue in 0..a0 {
      if n_arr[residue] == i128::MAX {
        continue;
      }
      for &aj in &nums[1..] {
        let new_val = n_arr[residue] + aj;
        let new_residue = (new_val as usize) % a0;
        if new_val < n_arr[new_residue] {
          n_arr[new_residue] = new_val;
          changed = true;
        }
      }
    }
  }

  // Frobenius number is max(n_arr) - a0
  let max_n = *n_arr.iter().max().unwrap();
  Ok(Expr::Integer(max_n - a0 as i128))
}

/// FrobeniusSolve[{a1, ..., an}, b] - All non-negative integer solutions
/// (x1, ..., xn) to a1*x1 + ... + an*xn == b.
///
/// FrobeniusSolve[{a1, ..., an}, b, m] returns at most m solutions (m a
/// positive integer or `All`).
pub fn frobenius_solve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Ok(unevaluated("FrobeniusSolve", args));
  }

  // First argument: a non-empty list of positive integers.
  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(unevaluated("FrobeniusSolve", args));
    }
  };
  let coef_err = || {
    crate::emit_message(&format!(
      "FrobeniusSolve::coef: The first argument {} of FrobeniusSolve should be a nonempty list of positive integers.",
      expr_to_string(&args[0])
    ));
    Ok::<Expr, InterpreterError>(unevaluated("FrobeniusSolve", args))
  };
  if items.is_empty() {
    return coef_err();
  }
  let mut coeffs: Vec<i128> = Vec::with_capacity(items.len());
  for item in items.iter() {
    match expr_to_i128(item) {
      Some(n) if n > 0 => coeffs.push(n),
      _ => return coef_err(),
    }
  }

  // Second argument: a non-negative integer target.
  let target = match expr_to_i128(&args[1]) {
    Some(n) if n >= 0 => n,
    _ => {
      return Ok(unevaluated("FrobeniusSolve", args));
    }
  };

  // Third argument: positive integer limit or `All`.
  let limit: Option<usize> = if args.len() == 3 {
    match &args[2] {
      Expr::Identifier(s) if s == "All" => None,
      e => match expr_to_i128(e) {
        Some(n) if n > 0 => Some(n as usize),
        _ => {
          crate::emit_message(&format!(
            "FrobeniusSolve::nsol: The number {} of requested solutions should be a positive integer.",
            expr_to_string(&args[2])
          ));
          return Ok(unevaluated("FrobeniusSolve", args));
        }
      },
    }
  } else {
    None
  };

  // Enumerate solutions in lexicographic order. The recursion fixes x1, then
  // x2, ..., so iterating x_i = 0, 1, ... yields lex-sorted solutions.
  fn enumerate(
    coeffs: &[i128],
    target: i128,
    current: &mut Vec<Expr>,
    results: &mut Vec<Expr>,
    limit: Option<usize>,
  ) -> bool {
    if limit.is_some_and(|l| results.len() >= l) {
      return true;
    }
    if coeffs.len() == 1 {
      let c = coeffs[0];
      if target % c == 0 {
        let k = target / c;
        current.push(Expr::Integer(k));
        results.push(Expr::List(current.clone().into()));
        current.pop();
      }
      return limit.is_some_and(|l| results.len() >= l);
    }
    let c0 = coeffs[0];
    let max_x = target / c0;
    for x in 0..=max_x {
      current.push(Expr::Integer(x));
      let stop =
        enumerate(&coeffs[1..], target - x * c0, current, results, limit);
      current.pop();
      if stop {
        return true;
      }
    }
    false
  }

  let mut results: Vec<Expr> = Vec::new();
  let mut current: Vec<Expr> = Vec::with_capacity(coeffs.len());
  enumerate(&coeffs, target, &mut current, &mut results, limit);
  Ok(Expr::List(results.into()))
}

/// PartitionsP[n] - Number of unrestricted partitions of the integer n
pub fn partitions_p_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PartitionsP expects exactly 1 argument".into(),
    ));
  }

  match expr_to_i128(&args[0]) {
    Some(n) => {
      if n < 0 {
        return Ok(Expr::Integer(0));
      }
      let n = n as usize;
      let result = partitions_p(n);
      Ok(bigint_to_expr(result))
    }
    None => Ok(unevaluated("PartitionsP", args)),
  }
}

/// Compute p(n) using dynamic programming
fn partitions_p(n: usize) -> BigInt {
  let mut dp = vec![BigInt::from(0); n + 1];
  dp[0] = BigInt::from(1);
  for k in 1..=n {
    for j in k..=n {
      dp[j] = &dp[j] + &dp[j - k];
    }
  }
  dp[n].clone()
}

/// PartitionsQ[n] - Number of partitions of n into distinct parts
pub fn partitions_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PartitionsQ expects exactly 1 argument".into(),
    ));
  }

  match expr_to_i128(&args[0]) {
    Some(n) => {
      if n < 0 {
        return Ok(Expr::Integer(0));
      }
      let n = n as usize;
      let result = partitions_q(n);
      Ok(bigint_to_expr(result))
    }
    None => Ok(unevaluated("PartitionsQ", args)),
  }
}

/// ArithmeticGeometricMean[a, b] - Computes the arithmetic-geometric mean
pub fn arithmetic_geometric_mean_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ArithmeticGeometricMean expects exactly 2 arguments".into(),
    ));
  }

  // At least one argument is an inexact (machine-real) number, so the result
  // is a machine real rather than an exact/symbolic expression.
  let inexact =
    matches!(&args[0], Expr::Real(_)) || matches!(&args[1], Expr::Real(_));

  // AGM[a, 0] = AGM[0, b] = 0, returning the zero argument so its type
  // (exact 0 vs. machine 0.) is preserved, matching wolframscript.
  let is_zero = |e: &Expr| {
    matches!(e, Expr::Integer(0)) || matches!(e, Expr::Real(x) if *x == 0.0)
  };
  if is_zero(&args[0]) {
    return Ok(args[0].clone());
  }
  if is_zero(&args[1]) {
    return Ok(args[1].clone());
  }

  // AGM[a, -a] = 0: when the two arguments are negatives of each other their
  // sum vanishes, and the geometric step Sqrt[a*(-a)] together with the mean 0
  // collapses the whole sequence to zero. This holds for exact, machine-real,
  // and purely symbolic negations (e.g. AGM[x, -x]). Detect it structurally by
  // testing whether a + b evaluates to zero, and preserve exactness (a
  // machine-real argument yields 0. rather than 0), matching wolframscript.
  if let Ok(sum) =
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![args[0].clone(), args[1].clone()].into(),
    })
    && is_zero(&sum)
  {
    return Ok(if inexact {
      Expr::Real(0.0)
    } else {
      Expr::Integer(0)
    });
  }

  // AGM[a, a] = a for equal symbolic atoms (bare symbols and constants such as
  // Pi). Compound equal arguments (e.g. x + 1, Sqrt[2]) stay unevaluated,
  // matching wolframscript; equal numbers are handled by the numeric path
  // below.
  if matches!(&args[0], Expr::Identifier(_) | Expr::Constant(_))
    && expr_to_string(&args[0]) == expr_to_string(&args[1])
  {
    return Ok(args[0].clone());
  }

  let (fa, fb) = (expr_to_f64(&args[0]), expr_to_f64(&args[1]));

  // AGM[a, a] = a. With a machine-real argument the result is a machine real.
  if let (Some(a), Some(b)) = (fa, fb)
    && a == b
  {
    if inexact {
      return Ok(Expr::Real(a));
    }
    return Ok(args[0].clone());
  }

  // Numeric evaluation when both are numeric and at least one is Real.
  // The AGM iteration converges to a real value only for arguments of the
  // same sign; AGM[-a, -b] = -AGM[a, b]. Mixed-sign reals yield a complex
  // value and are left for symbolic handling.
  if let (Some(a), Some(b)) = (fa, fb)
    && inexact
  {
    if a > 0.0 && b > 0.0 {
      return Ok(Expr::Real(agm(a, b)));
    }
    if a < 0.0 && b < 0.0 {
      return Ok(Expr::Real(-agm(-a, -b)));
    }
  }

  // Stay unevaluated for symbolic/exact inputs
  Ok(unevaluated("ArithmeticGeometricMean", args))
}

/// Compute AGM iteratively
fn agm(mut a: f64, mut b: f64) -> f64 {
  for _ in 0..100 {
    let a_new = (a + b) / 2.0;
    let b_new = (a * b).sqrt();
    if (a_new - b_new).abs() < 1e-15 * a_new.abs().max(1.0) {
      return a_new;
    }
    a = a_new;
    b = b_new;
  }
  (a + b) / 2.0
}

/// Compute q(n) - number of partitions into distinct parts using DP
/// Uses generating function: prod_{k=1}^{n} (1 + x^k)
fn partitions_q(n: usize) -> BigInt {
  let mut dp = vec![BigInt::from(0); n + 1];
  dp[0] = BigInt::from(1);
  for k in 1..=n {
    // Process in reverse to ensure each part k is used at most once
    for j in (k..=n).rev() {
      dp[j] = &dp[j] + &dp[j - k];
    }
  }
  dp[n].clone()
}

/// PrimeOmega[n] - number of prime factors with multiplicity
pub fn prime_omega_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // PrimeOmega is Listable: thread over a list of arguments.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|x| prime_omega_ast(std::slice::from_ref(x)))
      .collect();
    return Ok(Expr::List(results?.into()));
  }
  // wolframscript leaves PrimeOmega[0] unevaluated (0 has no prime
  // factorization); without this guard FactorInteger[0] = {{0, 1}} would
  // be miscounted as one prime factor.
  if matches!(&args[0], Expr::Integer(0)) {
    return Ok(unevaluated("PrimeOmega", args));
  }
  let factors = factor_integer_ast(args)?;
  if let Expr::List(ref pairs) = factors {
    let mut total: i128 = 0;
    for pair in pairs {
      if let Expr::List(pv) = pair
        && pv.len() == 2
        && let Expr::Integer(exp) = &pv[1]
      {
        // Skip the {-1, 1} and {1, 1} factors
        if let Expr::Integer(base) = &pv[0]
          && (*base == -1 || *base == 1)
        {
          continue;
        }
        total += *exp;
      }
    }
    Ok(Expr::Integer(total))
  } else {
    Ok(unevaluated("PrimeOmega", args))
  }
}

/// PrimeNu[n] - number of distinct prime factors
pub fn prime_nu_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // PrimeNu is Listable: thread over a list of arguments.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|x| prime_nu_ast(std::slice::from_ref(x)))
      .collect();
    return Ok(Expr::List(results?.into()));
  }
  // wolframscript leaves PrimeNu[0] unevaluated (0 has no prime
  // factorization); without this guard FactorInteger[0] = {{0, 1}} would
  // be miscounted as one distinct prime.
  if matches!(&args[0], Expr::Integer(0)) {
    return Ok(unevaluated("PrimeNu", args));
  }
  let factors = factor_integer_ast(args)?;
  if let Expr::List(ref pairs) = factors {
    let mut count: i128 = 0;
    for pair in pairs {
      if let Expr::List(pv) = pair
        && pv.len() == 2
      {
        // Skip the {-1, 1} and {1, 1} factors
        if let Expr::Integer(base) = &pv[0]
          && (*base == -1 || *base == 1)
        {
          continue;
        }
        count += 1;
      }
    }
    Ok(Expr::Integer(count))
  } else {
    Ok(unevaluated("PrimeNu", args))
  }
}

/// BitSet[n, k] - set the k-th bit (0-indexed) of n to 1: n | (1 << k)
/// BitSet[n, {k1, k2, ...}] - map over list of bit positions
pub fn bit_set_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(unevaluated("BitSet", args));
  }

  // Handle list of bit positions
  if let Expr::List(positions) = &args[1] {
    let results: Vec<Expr> = positions
      .iter()
      .map(|pos| {
        bit_set_ast(&[args[0].clone(), pos.clone()]).unwrap_or_else(|_| {
          Expr::FunctionCall {
            name: "BitSet".to_string(),
            args: vec![args[0].clone(), pos.clone()].into(),
          }
        })
      })
      .collect();
    return Ok(Expr::List(results.into()));
  }

  let n = match expr_to_bigint(&args[0]) {
    Some(n) => n,
    None => {
      return Ok(unevaluated("BitSet", args));
    }
  };

  let k = match &args[1] {
    Expr::Integer(k) if *k >= 0 => *k as u64,
    _ => {
      return Ok(unevaluated("BitSet", args));
    }
  };

  let bit = BigInt::from(1) << k;
  Ok(bigint_to_expr(n | bit))
}

/// BitClear[n, k] - clear the k-th bit (0-indexed) of n to 0: n & ~(1 << k)
/// BitClear[n, {k1, k2, ...}] - map over list of bit positions
pub fn bit_clear_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(unevaluated("BitClear", args));
  }

  if let Expr::List(positions) = &args[1] {
    let results: Vec<Expr> = positions
      .iter()
      .map(|pos| {
        bit_clear_ast(&[args[0].clone(), pos.clone()]).unwrap_or_else(|_| {
          Expr::FunctionCall {
            name: "BitClear".to_string(),
            args: vec![args[0].clone(), pos.clone()].into(),
          }
        })
      })
      .collect();
    return Ok(Expr::List(results.into()));
  }

  let n = match expr_to_bigint(&args[0]) {
    Some(n) => n,
    None => {
      return Ok(unevaluated("BitClear", args));
    }
  };

  let k = match &args[1] {
    Expr::Integer(k) if *k >= 0 => *k as u64,
    _ => {
      return Ok(unevaluated("BitClear", args));
    }
  };

  let bit = BigInt::from(1) << k;
  Ok(bigint_to_expr(n & !bit))
}

/// BitFlip[n, k] - flip the k-th bit of n: n XOR (1 << k)
/// For k >= 0: flip bit k (0-indexed from LSB)
/// For k < 0: flip bit (BitLength(n) + k) from LSB
pub fn bit_flip_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(unevaluated("BitFlip", args));
  }

  let n = match expr_to_bigint(&args[0]) {
    Some(n) => n,
    None => {
      return Ok(unevaluated("BitFlip", args));
    }
  };

  let k = match &args[1] {
    Expr::Integer(k) => *k,
    _ => {
      return Ok(unevaluated("BitFlip", args));
    }
  };

  let bit_pos = if k >= 0 {
    k as u64
  } else {
    // Negative index: count from MSB
    let bit_length = n.bits();
    if bit_length == 0 {
      return Ok(unevaluated("BitFlip", args));
    }
    let pos = bit_length as i128 + k;
    if pos < 0 {
      return Ok(unevaluated("BitFlip", args));
    }
    pos as u64
  };

  let bit = BigInt::from(1) << bit_pos;
  Ok(bigint_to_expr(n ^ bit))
}

/// Hyperfactorial[n] - Product of k^k for k=1..n
/// For negative integers: H(-m) = (-1)^(m*(m-1)/2) * Product[k^k, {k, 1, m-1}]
pub fn hyperfactorial_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if let Some(n) = expr_to_i128(&args[0]) {
    if n == 0 || n == 1 {
      return Ok(Expr::Integer(1));
    }
    if n > 0 {
      let mut result = BigInt::from(1);
      for k in 1..=n {
        let k_big = BigInt::from(k);
        result *= num_traits::pow::Pow::pow(&k_big, k as u64);
      }
      return Ok(bigint_to_expr(result));
    }
    // Negative integers: H(-m) = (-1)^(m*(m-1)/2) * Product[k^k, {k, 1, m-1}]
    let m = (-n) as u64;
    let mut product = BigInt::from(1);
    for k in 1..m {
      let k_big = BigInt::from(k);
      product *= num_traits::pow::Pow::pow(&k_big, k);
    }
    let sign_exp = (m * (m - 1) / 2) % 2;
    if sign_exp == 1 {
      product = -product;
    }
    return Ok(bigint_to_expr(product));
  }
  Ok(unevaluated("Hyperfactorial", args))
}

/// DeBruijnSequence[alphabet, n] - De Bruijn sequence
/// Returns a cyclic sequence of minimum length in which every possible
/// subsequence of length n from the given alphabet appears exactly once.
pub fn debruijn_sequence_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let alphabet = match &args[0] {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(unevaluated("DeBruijnSequence", args));
    }
  };
  let n = match expr_to_i128(&args[1]) {
    Some(n) if n >= 1 => n as usize,
    _ => {
      return Ok(unevaluated("DeBruijnSequence", args));
    }
  };

  let k = alphabet.len();
  if k == 0 {
    return Ok(Expr::List(vec![].into()));
  }
  if k == 1 {
    return Ok(Expr::List(vec![alphabet[0].clone()].into()));
  }

  // Martin's algorithm for generating De Bruijn sequences
  let mut sequence: Vec<usize> = Vec::new();
  let mut a = vec![0usize; k * n + 1];

  fn db(
    t: usize,
    p: usize,
    n: usize,
    k: usize,
    a: &mut Vec<usize>,
    seq: &mut Vec<usize>,
  ) {
    if t > n {
      if n.is_multiple_of(p) {
        for i in 1..=p {
          seq.push(a[i]);
        }
      }
    } else {
      a[t] = a[t - p];
      db(t + 1, p, n, k, a, seq);
      for j in (a[t - p] + 1)..k {
        a[t] = j;
        db(t + 1, t, n, k, a, seq);
      }
    }
  }

  db(1, 1, n, k, &mut a, &mut sequence);

  let result: Vec<Expr> =
    sequence.iter().map(|&idx| alphabet[idx].clone()).collect();
  Ok(Expr::List(result.into()))
}

/// BellY[n, k, {x1, x2, ...}] - Partial Bell polynomial (Bell Y polynomial)
/// B_{n,k}(x1, x2, ..., x_{n-k+1})
/// Uses the formula: sum over all partitions of n into k parts
pub fn bell_y_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n as usize,
    _ => {
      return Ok(unevaluated("BellY", args));
    }
  };
  let k = match expr_to_i128(&args[1]) {
    Some(k) if k >= 0 => k as usize,
    _ => {
      return Ok(unevaluated("BellY", args));
    }
  };
  let xs = match &args[2] {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(unevaluated("BellY", args));
    }
  };

  // Base cases
  if n == 0 && k == 0 {
    return Ok(Expr::Integer(1));
  }
  if k == 0 || n == 0 {
    return Ok(Expr::Integer(0));
  }
  if k > n {
    return Ok(Expr::Integer(0));
  }

  // Generate all partitions of n into exactly k parts,
  // where each part is between 1 and n-k+1
  // A partition here is (j1, j2, ..., j_{n-k+1}) where j_i >= 0,
  // sum(i * j_i) = n, sum(j_i) = k
  let max_part = n - k + 1;
  let mut terms: Vec<Expr> = Vec::new();

  // Enumerate partitions using recursive approach
  let mut partition = vec![0usize; max_part + 1]; // partition[i] = count of parts equal to i
  bell_y_enumerate(n, k, max_part, &mut partition, &xs, &mut terms)?;

  if terms.is_empty() {
    return Ok(Expr::Integer(0));
  }
  if terms.len() == 1 {
    return Ok(terms.into_iter().next().unwrap());
  }
  crate::evaluator::evaluate_function_call_ast("Plus", &terms)
}

/// Recursively enumerate partitions for BellY
fn bell_y_enumerate(
  remaining_sum: usize,
  remaining_parts: usize,
  max_val: usize,
  partition: &mut Vec<usize>,
  xs: &[Expr],
  terms: &mut Vec<Expr>,
) -> Result<(), InterpreterError> {
  if remaining_parts == 0 {
    if remaining_sum == 0 {
      // Compute the term: n! / (prod(j_i!) * prod((i!)^j_i)) * prod(x_i^j_i)
      // The coefficient is: n! / (j1! * j2! * ... * (1!)^j1 * (2!)^j2 * ...)
      // But for BellY, the formula is:
      // (n! / prod(j_i! * i!^j_i)) * prod(x_i^j_i)
      // where the sum is over partitions satisfying sum(i*j_i)=n, sum(j_i)=k
      let n: usize = partition
        .iter()
        .enumerate()
        .map(|(i, &count)| i * count)
        .sum();
      let mut coeff_num = BigInt::from(1);
      // n!
      for i in 1..=n {
        coeff_num *= BigInt::from(i);
      }
      let mut coeff_den = BigInt::from(1);
      let mut factors: Vec<Expr> = Vec::new();
      for (i, &j_i) in partition.iter().enumerate() {
        if i == 0 || j_i == 0 {
          continue;
        }
        // j_i!
        for m in 1..=j_i {
          coeff_den *= BigInt::from(m);
        }
        // (i!)^j_i
        let mut i_fact = BigInt::from(1);
        for m in 1..=i {
          i_fact *= BigInt::from(m);
        }
        for _ in 0..j_i {
          coeff_den *= &i_fact;
        }
        // x_i^j_i (x is 0-indexed, x[0] = x_1 which corresponds to i=1)
        if i - 1 < xs.len() {
          if j_i == 1 {
            factors.push(xs[i - 1].clone());
          } else {
            factors.push(Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![xs[i - 1].clone(), Expr::Integer(j_i as i128)].into(),
            });
          }
        }
      }
      let coeff = coeff_num / coeff_den;
      let coeff_expr = bigint_to_expr(coeff.clone());
      let term = if coeff == BigInt::from(1) && !factors.is_empty() {
        if factors.len() == 1 {
          factors[0].clone()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: factors.into(),
          }
        }
      } else if factors.is_empty() {
        coeff_expr
      } else {
        let mut all = vec![coeff_expr];
        all.extend(factors);
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: all.into(),
        }
      };
      let evaluated =
        crate::evaluator::evaluate_function_call_ast("Times", &[term])?;
      terms.push(evaluated);
    }
    return Ok(());
  }

  if max_val == 0 {
    return Ok(());
  }

  // Try each count for parts of size max_val
  let max_count = std::cmp::min(remaining_parts, remaining_sum / max_val);
  for count in (0..=max_count).rev() {
    partition[max_val] = count;
    bell_y_enumerate(
      remaining_sum - max_val * count,
      remaining_parts - count,
      max_val - 1,
      partition,
      xs,
      terms,
    )?;
  }
  partition[max_val] = 0;
  Ok(())
}

/// FiniteGroupCount[n] - Number of groups of order n
/// Uses a lookup table for orders 1-2047
pub fn finite_group_count_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 1 => n as usize,
    // There are no groups of order 0, so Wolfram returns 0.
    Some(0) => return Ok(bigint_to_expr(BigInt::from(0))),
    Some(_) => {
      // Negative: return unevaluated (Wolfram returns error message)
      return Ok(unevaluated("FiniteGroupCount", args));
    }
    None => {
      return Ok(unevaluated("FiniteGroupCount", args));
    }
  };

  if n <= FINITE_GROUP_COUNT_TABLE.len() {
    return Ok(bigint_to_expr(BigInt::from(
      FINITE_GROUP_COUNT_TABLE[n - 1],
    )));
  }

  // Beyond lookup table range, return unevaluated
  Ok(unevaluated("FiniteGroupCount", args))
}

/// FiniteAbelianGroupCount[n] - Number of abelian groups of order n
/// Equals the product of partition numbers of the exponents in the prime factorization
pub fn finite_abelian_group_count_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 1 => n,
    // There are no abelian groups of order 0, so Wolfram returns 0.
    Some(0) => return Ok(bigint_to_expr(BigInt::from(0))),
    Some(_) => {
      return Ok(unevaluated("FiniteAbelianGroupCount", args));
    }
    None => {
      return Ok(unevaluated("FiniteAbelianGroupCount", args));
    }
  };

  if n == 1 {
    return Ok(Expr::Integer(1));
  }

  // Factor n and compute product of partition counts of exponents
  let mut num = n.unsigned_abs();
  let mut result = BigInt::from(1);

  // Factor out 2
  let mut exp = 0usize;
  while num % 2 == 0 {
    exp += 1;
    num /= 2;
  }
  if exp > 0 {
    result *= partitions_p(exp);
  }

  // Factor out odd primes
  let mut i: u128 = 3;
  while i * i <= num {
    let mut exp = 0usize;
    while num % i == 0 {
      exp += 1;
      num /= i;
    }
    if exp > 0 {
      result *= partitions_p(exp);
    }
    i += 2;
  }
  if num > 1 {
    // Remaining prime factor with exponent 1, partitions_p(1) = 1
    // result *= 1; -- no-op
  }

  Ok(bigint_to_expr(result))
}

// Lookup table for FiniteGroupCount[n] for n = 1..2047
// Source: OEIS A000001
const FINITE_GROUP_COUNT_TABLE: &[u64] = &[
  1,
  1,
  1,
  2,
  1,
  2,
  1,
  5,
  2,
  2,
  1,
  5,
  1,
  2,
  1,
  14,
  1,
  5,
  1,
  5,
  2,
  2,
  1,
  15,
  2,
  2,
  5,
  4,
  1,
  4,
  1,
  51,
  1,
  2,
  1,
  14,
  1,
  2,
  2,
  14,
  1,
  6,
  1,
  4,
  2,
  2,
  1,
  52,
  2,
  5,
  1,
  5,
  1,
  15,
  2,
  13,
  2,
  2,
  1,
  13,
  1,
  2,
  4,
  267,
  1,
  4,
  1,
  5,
  1,
  4,
  1,
  50,
  1,
  2,
  3,
  4,
  1,
  6,
  1,
  52,
  15,
  2,
  1,
  15,
  1,
  2,
  1,
  12,
  1,
  10,
  1,
  4,
  2,
  2,
  1,
  231,
  1,
  5,
  2,
  16,
  1,
  4,
  1,
  14,
  2,
  2,
  1,
  45,
  1,
  6,
  2,
  43,
  1,
  6,
  1,
  5,
  4,
  2,
  1,
  47,
  2,
  2,
  1,
  4,
  5,
  16,
  1,
  2328,
  2,
  4,
  1,
  10,
  1,
  2,
  5,
  15,
  1,
  4,
  1,
  11,
  1,
  2,
  1,
  197,
  1,
  2,
  6,
  5,
  1,
  13,
  1,
  12,
  2,
  4,
  2,
  18,
  1,
  2,
  1,
  238,
  1,
  55,
  1,
  5,
  2,
  2,
  1,
  57,
  2,
  4,
  5,
  4,
  1,
  4,
  2,
  42,
  1,
  2,
  1,
  37,
  1,
  4,
  2,
  12,
  1,
  6,
  1,
  4,
  13,
  4,
  1,
  1543,
  1,
  2,
  2,
  12,
  1,
  10,
  1,
  52,
  2,
  2,
  2,
  12,
  2,
  2,
  2,
  51,
  1,
  12,
  1,
  5,
  1,
  2,
  1,
  177,
  1,
  2,
  2,
  15,
  1,
  6,
  1,
  197,
  6,
  2,
  1,
  15,
  1,
  4,
  2,
  14,
  1,
  16,
  1,
  4,
  2,
  4,
  1,
  208,
  1,
  5,
  67,
  5,
  2,
  4,
  1,
  12,
  1,
  15,
  1,
  46,
  2,
  2,
  1,
  56092,
  1,
  6,
  1,
  15,
  2,
  2,
  1,
  39,
  1,
  4,
  1,
  4,
  1,
  30,
  1,
  54,
  5,
  2,
  4,
  10,
  1,
  2,
  4,
  40,
  1,
  4,
  1,
  4,
  2,
  4,
  1,
  1045,
  2,
  4,
  2,
  5,
  1,
  23,
  1,
  14,
  5,
  2,
  1,
  49,
  2,
  2,
  1,
  42,
  2,
  10,
  1,
  9,
  2,
  6,
  1,
  61,
  1,
  2,
  4,
  4,
  1,
  4,
  1,
  1640,
  1,
  4,
  1,
  176,
  2,
  2,
  2,
  15,
  1,
  12,
  1,
  4,
  5,
  2,
  1,
  228,
  1,
  5,
  1,
  15,
  1,
  18,
  5,
  12,
  1,
  2,
  1,
  12,
  1,
  10,
  14,
  195,
  1,
  4,
  2,
  5,
  2,
  2,
  1,
  162,
  2,
  2,
  3,
  11,
  1,
  6,
  1,
  42,
  2,
  4,
  1,
  15,
  1,
  4,
  7,
  12,
  1,
  60,
  1,
  11,
  2,
  2,
  1,
  20169,
  2,
  2,
  4,
  5,
  1,
  12,
  1,
  44,
  1,
  2,
  1,
  30,
  1,
  2,
  5,
  221,
  1,
  6,
  1,
  5,
  16,
  6,
  1,
  46,
  1,
  6,
  1,
  4,
  1,
  10,
  1,
  235,
  2,
  4,
  1,
  41,
  1,
  2,
  2,
  14,
  2,
  4,
  1,
  4,
  2,
  4,
  1,
  775,
  1,
  4,
  1,
  5,
  1,
  6,
  1,
  51,
  13,
  4,
  1,
  18,
  1,
  2,
  1,
  1396,
  1,
  34,
  1,
  5,
  2,
  2,
  1,
  54,
  1,
  2,
  5,
  11,
  1,
  12,
  1,
  51,
  4,
  2,
  1,
  55,
  1,
  4,
  2,
  12,
  1,
  6,
  2,
  11,
  2,
  2,
  1,
  1213,
  1,
  2,
  2,
  12,
  1,
  261,
  1,
  14,
  2,
  10,
  1,
  12,
  1,
  4,
  4,
  42,
  2,
  4,
  1,
  56,
  1,
  2,
  2,
  195,
  2,
  6,
  6,
  4,
  1,
  8,
  1,
  10494213,
  15,
  2,
  1,
  15,
  1,
  4,
  1,
  49,
  1,
  10,
  1,
  4,
  6,
  2,
  1,
  170,
  2,
  4,
  2,
  9,
  1,
  4,
  1,
  12,
  1,
  2,
  2,
  119,
  1,
  2,
  2,
  246,
  1,
  24,
  1,
  5,
  4,
  16,
  1,
  39,
  1,
  2,
  2,
  4,
  1,
  16,
  1,
  180,
  1,
  2,
  1,
  10,
  1,
  2,
  49,
  12,
  1,
  12,
  1,
  11,
  1,
  4,
  2,
  8681,
  1,
  5,
  2,
  15,
  1,
  6,
  1,
  15,
  4,
  2,
  1,
  66,
  1,
  4,
  1,
  51,
  1,
  30,
  1,
  5,
  2,
  4,
  1,
  205,
  1,
  6,
  4,
  4,
  7,
  4,
  1,
  195,
  3,
  6,
  1,
  36,
  1,
  2,
  2,
  35,
  1,
  6,
  1,
  15,
  5,
  2,
  1,
  260,
  15,
  2,
  2,
  5,
  1,
  32,
  1,
  12,
  2,
  2,
  1,
  12,
  2,
  4,
  2,
  21541,
  1,
  4,
  1,
  9,
  2,
  4,
  1,
  757,
  1,
  10,
  5,
  4,
  1,
  6,
  2,
  53,
  5,
  4,
  1,
  40,
  1,
  2,
  2,
  12,
  1,
  18,
  1,
  4,
  2,
  4,
  1,
  1280,
  1,
  2,
  17,
  16,
  1,
  4,
  1,
  53,
  1,
  4,
  1,
  51,
  1,
  15,
  2,
  42,
  2,
  8,
  1,
  5,
  4,
  2,
  1,
  44,
  1,
  2,
  1,
  36,
  1,
  62,
  1,
  1387,
  1,
  2,
  1,
  10,
  1,
  6,
  4,
  15,
  1,
  12,
  2,
  4,
  1,
  2,
  1,
  840,
  1,
  5,
  2,
  5,
  2,
  13,
  1,
  40,
  504,
  4,
  1,
  18,
  1,
  2,
  6,
  195,
  2,
  10,
  1,
  15,
  5,
  4,
  1,
  54,
  1,
  2,
  2,
  11,
  1,
  39,
  1,
  42,
  1,
  4,
  2,
  189,
  1,
  2,
  2,
  39,
  1,
  6,
  1,
  4,
  2,
  2,
  1,
  1090235,
  1,
  12,
  1,
  5,
  1,
  16,
  4,
  15,
  5,
  2,
  1,
  53,
  1,
  4,
  5,
  172,
  1,
  4,
  1,
  5,
  1,
  4,
  2,
  137,
  1,
  2,
  1,
  4,
  1,
  24,
  1,
  1211,
  2,
  2,
  1,
  15,
  1,
  4,
  1,
  14,
  1,
  113,
  1,
  16,
  2,
  4,
  1,
  205,
  1,
  2,
  11,
  20,
  1,
  4,
  1,
  12,
  5,
  4,
  1,
  30,
  1,
  4,
  2,
  1630,
  2,
  6,
  1,
  9,
  13,
  2,
  1,
  186,
  2,
  2,
  1,
  4,
  2,
  10,
  2,
  51,
  2,
  10,
  1,
  10,
  1,
  4,
  5,
  12,
  1,
  12,
  1,
  11,
  2,
  2,
  1,
  4725,
  1,
  2,
  3,
  9,
  1,
  8,
  1,
  14,
  4,
  4,
  5,
  18,
  1,
  2,
  1,
  221,
  1,
  68,
  1,
  15,
  1,
  2,
  1,
  61,
  2,
  4,
  15,
  4,
  1,
  4,
  1,
  19349,
  2,
  2,
  1,
  150,
  1,
  4,
  7,
  15,
  2,
  6,
  1,
  4,
  2,
  8,
  1,
  222,
  1,
  2,
  4,
  5,
  1,
  30,
  1,
  39,
  2,
  2,
  1,
  34,
  2,
  2,
  4,
  235,
  1,
  18,
  2,
  5,
  1,
  2,
  2,
  222,
  1,
  4,
  2,
  11,
  1,
  6,
  1,
  42,
  13,
  4,
  1,
  15,
  1,
  10,
  1,
  42,
  1,
  10,
  2,
  4,
  1,
  2,
  1,
  11394,
  2,
  4,
  2,
  5,
  1,
  12,
  1,
  42,
  2,
  4,
  1,
  900,
  1,
  2,
  6,
  51,
  1,
  6,
  2,
  34,
  5,
  2,
  1,
  46,
  1,
  4,
  2,
  11,
  1,
  30,
  1,
  196,
  2,
  6,
  1,
  10,
  1,
  2,
  15,
  199,
  1,
  4,
  1,
  4,
  2,
  2,
  1,
  954,
  1,
  6,
  2,
  13,
  1,
  23,
  2,
  12,
  2,
  2,
  1,
  37,
  1,
  4,
  2,
  49487365422,
  4,
  66,
  2,
  5,
  19,
  4,
  1,
  54,
  1,
  4,
  2,
  11,
  1,
  4,
  1,
  231,
  1,
  2,
  1,
  36,
  2,
  2,
  2,
  12,
  1,
  40,
  1,
  4,
  51,
  4,
  2,
  1028,
  1,
  5,
  1,
  15,
  1,
  10,
  1,
  35,
  2,
  4,
  1,
  12,
  1,
  4,
  4,
  42,
  1,
  4,
  2,
  5,
  1,
  10,
  1,
  583,
  2,
  2,
  6,
  4,
  2,
  6,
  1,
  1681,
  6,
  4,
  1,
  77,
  1,
  2,
  2,
  15,
  1,
  16,
  1,
  51,
  2,
  4,
  1,
  170,
  1,
  4,
  5,
  5,
  1,
  12,
  1,
  12,
  2,
  2,
  1,
  46,
  1,
  4,
  2,
  1092,
  1,
  8,
  1,
  5,
  14,
  2,
  2,
  39,
  1,
  4,
  2,
  4,
  1,
  254,
  1,
  42,
  2,
  2,
  1,
  41,
  1,
  2,
  5,
  39,
  1,
  4,
  1,
  11,
  1,
  10,
  1,
  157877,
  1,
  2,
  4,
  16,
  1,
  6,
  1,
  49,
  13,
  4,
  1,
  18,
  1,
  4,
  1,
  53,
  1,
  32,
  1,
  5,
  1,
  2,
  2,
  279,
  1,
  4,
  2,
  11,
  1,
  4,
  3,
  235,
  2,
  2,
  1,
  99,
  1,
  8,
  2,
  14,
  1,
  6,
  1,
  11,
  14,
  2,
  1,
  1040,
  1,
  2,
  1,
  13,
  2,
  16,
  1,
  12,
  5,
  27,
  1,
  12,
  1,
  2,
  69,
  1387,
  1,
  16,
  1,
  20,
  2,
  4,
  1,
  164,
  4,
  2,
  2,
  4,
  1,
  12,
  1,
  153,
  2,
  2,
  1,
  15,
  1,
  2,
  2,
  51,
  1,
  30,
  1,
  4,
  1,
  4,
  1,
  1460,
  1,
  55,
  4,
  5,
  1,
  12,
  2,
  14,
  1,
  4,
  1,
  131,
  1,
  2,
  2,
  42,
  3,
  6,
  1,
  5,
  5,
  4,
  1,
  44,
  1,
  10,
  3,
  11,
  1,
  10,
  1,
  1116461,
  5,
  2,
  1,
  10,
  1,
  2,
  4,
  35,
  1,
  12,
  1,
  11,
  1,
  2,
  1,
  3609,
  1,
  4,
  2,
  50,
  1,
  24,
  1,
  12,
  2,
  2,
  1,
  18,
  1,
  6,
  2,
  244,
  1,
  18,
  1,
  9,
  2,
  2,
  1,
  181,
  1,
  2,
  51,
  4,
  2,
  12,
  1,
  42,
  1,
  8,
  5,
  61,
  1,
  4,
  1,
  12,
  1,
  6,
  1,
  11,
  2,
  4,
  1,
  11720,
  1,
  2,
  1,
  5,
  1,
  112,
  1,
  52,
  1,
  2,
  2,
  12,
  1,
  4,
  4,
  245,
  1,
  4,
  1,
  9,
  5,
  2,
  1,
  211,
  2,
  4,
  2,
  38,
  1,
  6,
  15,
  195,
  15,
  6,
  2,
  29,
  1,
  2,
  1,
  14,
  1,
  32,
  1,
  4,
  2,
  4,
  1,
  198,
  1,
  4,
  8,
  5,
  1,
  4,
  1,
  153,
  1,
  2,
  1,
  227,
  2,
  4,
  5,
  19324,
  1,
  8,
  1,
  5,
  4,
  4,
  1,
  39,
  1,
  2,
  2,
  15,
  4,
  16,
  1,
  53,
  6,
  4,
  1,
  40,
  1,
  12,
  5,
  12,
  1,
  4,
  2,
  4,
  1,
  2,
  1,
  5958,
  1,
  4,
  5,
  12,
  2,
  6,
  1,
  14,
  4,
  10,
  1,
  40,
  1,
  2,
  2,
  179,
  1,
  1798,
  1,
  15,
  2,
  4,
  1,
  61,
  1,
  2,
  5,
  4,
  1,
  46,
  1,
  1387,
  1,
  6,
  2,
  36,
  2,
  2,
  1,
  49,
  1,
  24,
  1,
  11,
  10,
  2,
  1,
  222,
  1,
  4,
  3,
  5,
  1,
  10,
  1,
  41,
  2,
  4,
  1,
  174,
  1,
  2,
  2,
  195,
  2,
  4,
  1,
  15,
  1,
  6,
  1,
  889,
  1,
  2,
  2,
  4,
  1,
  12,
  2,
  178,
  13,
  2,
  1,
  15,
  4,
  4,
  1,
  12,
  1,
  20,
  1,
  4,
  5,
  4,
  1,
  408641062,
  1,
  2,
  60,
  36,
  1,
  4,
  1,
  15,
  2,
  2,
  1,
  46,
  1,
  16,
  1,
  54,
  1,
  24,
  2,
  5,
  2,
  4,
  1,
  221,
  1,
  4,
  1,
  11,
  1,
  30,
  1,
  928,
  2,
  4,
  1,
  10,
  2,
  2,
  13,
  14,
  1,
  4,
  1,
  11,
  2,
  6,
  1,
  697,
  1,
  4,
  3,
  5,
  1,
  8,
  1,
  12,
  5,
  2,
  2,
  64,
  1,
  4,
  2,
  10281,
  1,
  10,
  1,
  5,
  1,
  4,
  1,
  54,
  1,
  8,
  2,
  11,
  1,
  4,
  1,
  51,
  6,
  2,
  1,
  477,
  1,
  2,
  2,
  56,
  5,
  6,
  1,
  11,
  5,
  4,
  1,
  1213,
  1,
  4,
  2,
  5,
  1,
  72,
  1,
  68,
  2,
  2,
  1,
  12,
  1,
  2,
  13,
  42,
  1,
  38,
  1,
  9,
  2,
  2,
  2,
  137,
  1,
  2,
  5,
  11,
  1,
  6,
  1,
  21507,
  5,
  10,
  1,
  15,
  1,
  4,
  1,
  34,
  2,
  60,
  2,
  4,
  5,
  2,
  1,
  1005,
  2,
  5,
  2,
  5,
  1,
  4,
  1,
  12,
  1,
  10,
  1,
  30,
  1,
  10,
  1,
  235,
  1,
  6,
  1,
  50,
  309,
  4,
  2,
  39,
  7,
  2,
  1,
  11,
  1,
  36,
  2,
  42,
  2,
  2,
  5,
  40,
  1,
  2,
  2,
  39,
  1,
  12,
  1,
  4,
  3,
  2,
  1,
  47937,
  1,
  4,
  2,
  5,
  1,
  13,
  1,
  35,
  4,
  4,
  1,
  37,
  1,
  4,
  2,
  51,
  1,
  16,
  1,
  9,
  1,
  30,
  2,
  64,
  1,
  2,
  14,
  4,
  1,
  4,
  1,
  1285,
  1,
  2,
  1,
  228,
  1,
  2,
  5,
  53,
  1,
  8,
  2,
  4,
  2,
  2,
  4,
  260,
  1,
  6,
  1,
  15,
  1,
  110,
  1,
  12,
  2,
  4,
  1,
  12,
  1,
  4,
  5,
  1083553,
  1,
  12,
  1,
  5,
  1,
  4,
  1,
  749,
  1,
  4,
  2,
  11,
  3,
  30,
  1,
  54,
  13,
  6,
  1,
  15,
  2,
  2,
  9,
  12,
  1,
  10,
  1,
  35,
  2,
  2,
  1,
  1264,
  2,
  4,
  6,
  5,
  1,
  18,
  1,
  14,
  2,
  4,
  1,
  117,
  1,
  2,
  2,
  178,
  1,
  6,
  1,
  5,
  4,
  4,
  1,
  162,
  2,
  10,
  1,
  4,
  1,
  16,
  1,
  1630,
  2,
  2,
  2,
  56,
  1,
  10,
  15,
  15,
  1,
  4,
  1,
  4,
  2,
  12,
  1,
  1096,
  1,
  2,
  21,
  9,
  1,
  6,
  1,
  39,
  5,
  2,
  1,
  18,
  1,
  4,
  2,
  195,
  1,
  120,
  1,
  9,
  2,
  2,
  1,
  54,
  1,
  4,
  4,
  36,
  1,
  4,
  1,
  186,
  2,
  2,
  1,
  36,
  1,
  6,
  15,
  12,
  1,
  8,
  1,
  4,
  5,
  4,
  1,
  241004,
  1,
  5,
  1,
  15,
  4,
  10,
  1,
  15,
  2,
  4,
  1,
  34,
  1,
  2,
  4,
  167,
  1,
  12,
  1,
  15,
  1,
  2,
  1,
  3973,
  1,
  4,
  1,
  4,
  1,
  40,
  1,
  235,
  11,
  2,
  1,
  15,
  1,
  6,
  1,
  144,
  1,
  18,
  1,
  4,
  2,
  2,
  2,
  203,
  1,
  4,
  15,
  15,
  1,
  12,
  2,
  39,
  1,
  4,
  1,
  120,
  1,
  2,
  2,
  1388,
  1,
  6,
  1,
  13,
  4,
  4,
  1,
  39,
  1,
  2,
  5,
  4,
  1,
  66,
  1,
  963,
  1,
  8,
  1,
  10,
  2,
  4,
  4,
  12,
  2,
  12,
  1,
  4,
  2,
  4,
  2,
  6538,
  1,
  2,
  2,
  20,
  1,
  6,
  2,
  46,
  63,
  2,
  1,
  88,
  1,
  12,
  1,
  42,
  1,
  10,
  2,
  5,
  5,
  2,
  1,
  175,
  2,
  2,
  2,
  11,
  1,
  12,
  1,
];

/// Return 2*j as i128 if `expr` is a non-negative integer or half-integer,
/// else None. Supports `Rational[n, 2]` and `Integer`.
fn two_half_int(expr: &Expr) -> Option<i128> {
  match expr_to_rational(expr) {
    Some((n, 1)) => Some(n.checked_mul(2)?),
    Some((n, 2)) => Some(n),
    _ => None,
  }
}

/// ThreeJSymbol[{j1, m1}, {j2, m2}, {j3, m3}]
///
/// Currently implements only the degenerate cases that evaluate to 0:
/// - m1 + m2 + m3 ≠ 0
/// - |m_i| > j_i for any i
/// - |j1 - j2| > j3 or j3 > j1 + j2 (triangle inequality)
///
/// For all other (valid) cases the call is returned unchanged.
pub fn three_j_symbol_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "ThreeJSymbol expects exactly 3 arguments".into(),
    ));
  }
  let unchanged = || unevaluated("ThreeJSymbol", args);
  let mut two_j = [0i128; 3];
  let mut two_m = [0i128; 3];
  for (i, arg) in args.iter().enumerate() {
    let Expr::List(items) = arg else {
      return Ok(unchanged());
    };
    if items.len() != 2 {
      return Ok(unchanged());
    }
    let (Some(j2), Some(m2)) =
      (two_half_int(&items[0]), two_half_int_signed(&items[1]))
    else {
      return Ok(unchanged());
    };
    if j2 < 0 {
      return Ok(unchanged());
    }
    two_j[i] = j2;
    two_m[i] = m2;
  }
  // m1 + m2 + m3 must be 0
  if two_m[0] + two_m[1] + two_m[2] != 0 {
    return Ok(Expr::Integer(0));
  }
  // |m_i| <= j_i
  for i in 0..3 {
    if two_m[i].abs() > two_j[i] {
      return Ok(Expr::Integer(0));
    }
    // j_i - m_i must be an integer, i.e. 2j_i and 2m_i have the same parity
    if (two_j[i] - two_m[i]).rem_euclid(2) != 0 {
      return Ok(Expr::Integer(0));
    }
  }
  // Triangle inequality: |j1 - j2| <= j3 <= j1 + j2
  let (j1, j2, j3) = (two_j[0], two_j[1], two_j[2]);
  if j3 < (j1 - j2).abs() || j3 > j1 + j2 {
    return Ok(Expr::Integer(0));
  }
  // j1 + j2 + j3 must be an integer (i.e. 2(j1+j2+j3) even)
  if (j1 + j2 + j3).rem_euclid(2) != 0 {
    return Ok(Expr::Integer(0));
  }
  let (m1, m2, m3) = (two_m[0], two_m[1], two_m[2]);

  // Wigner 3j Racah formula:
  //   (j1 m1, j2 m2, j3 m3) = (-1)^(j1-j2-m3)
  //     · Sqrt[Δ²(j1,j2,j3) · Π_i (j_i+m_i)!·(j_i-m_i)!]
  //     · Σ_t (-1)^t / [t!·(j1+j2-j3-t)!·(j1-m1-t)!·(j2+m2-t)!
  //                      · (j3-j2+m1+t)!·(j3-j1-m2+t)!]
  // Everything below is computed in integer 2j/2m units; halve the
  // sums/differences before plugging into factorial.
  fn fact(n: i128) -> BigInt {
    let mut acc = BigInt::from(1);
    for i in 2..=n {
      acc *= i;
    }
    acc
  }
  // Δ²(j1, j2, j3): same form as the 6j helper (uses 2j units).
  let n1 = (j1 + j2 - j3) / 2;
  let n2 = (j1 - j2 + j3) / 2;
  let n3 = (-j1 + j2 + j3) / 2;
  let nd = (j1 + j2 + j3) / 2 + 1;
  let mut radicand_num = fact(n1) * fact(n2) * fact(n3);
  let radicand_den = fact(nd);
  // Π_i (j_i + m_i)!·(j_i - m_i)! using 2j/2m units (every sum/difference
  // is even because (2j_i, 2m_i) share parity).
  for i in 0..3 {
    let p = (two_j[i] + two_m[i]) / 2;
    let m = (two_j[i] - two_m[i]) / 2;
    radicand_num *= fact(p);
    radicand_num *= fact(m);
  }

  // Σ_t. t ranges over integers because every (j ± m), (j1+j2-j3) etc.
  // is an integer here.
  let t_min = [0, (j2 - j3 - m1) / 2, (j1 + m2 - j3) / 2]
    .into_iter()
    .max()
    .unwrap();
  let t_max = [(j1 + j2 - j3) / 2, (j1 - m1) / 2, (j2 + m2) / 2]
    .into_iter()
    .min()
    .unwrap();
  let mut sum_num = BigInt::from(0);
  let mut sum_den = BigInt::from(1);
  let mut t = t_min;
  while t <= t_max {
    let denoms = [
      t,
      (j1 + j2 - j3) / 2 - t,
      (j1 - m1) / 2 - t,
      (j2 + m2) / 2 - t,
      (j3 - j2 + m1) / 2 + t,
      (j3 - j1 - m2) / 2 + t,
    ];
    if denoms.iter().any(|d| *d < 0) {
      t += 1;
      continue;
    }
    let mut term_den = BigInt::from(1);
    for d in denoms {
      term_den *= fact(d);
    }
    let term_num = BigInt::from(1);
    let sign = if t.rem_euclid(2) == 0 { 1 } else { -1 };
    let new_num =
      &sum_num * &term_den + BigInt::from(sign) * &term_num * &sum_den;
    let new_den = &sum_den * &term_den;
    sum_num = new_num;
    sum_den = new_den;
    t += 1;
  }

  // Sign factor: (-1)^(j1 - j2 - m3). Compute in 2j/2m units; the
  // exponent is an integer because the parity check ensures
  // 2(j1 - j2 - m3) is even.
  let outer_sign_exp = (j1 - j2 - m3) / 2;
  let outer_sign: i128 = if outer_sign_exp.rem_euclid(2) == 0 {
    1
  } else {
    -1
  };
  // Combine the outer sign with the sum's coefficient.
  sum_num *= outer_sign;

  // Split radicand into a perfect-square scalar (pulled outside) and the
  // remaining square-free residue (kept under Sqrt).
  let (out_n, in_n) = extract_perfect_square_bigint(&radicand_num);
  let (out_d, in_d) = extract_perfect_square_bigint(&radicand_den);
  let coeff_num = sum_num * &out_n;
  let coeff_den = &sum_den * &out_d;
  let coeff_expr = bigint_rational_to_expr(coeff_num, coeff_den);
  let radicand_expr = if in_d == BigInt::from(1) {
    bigint_to_expr(in_n)
  } else {
    crate::evaluator::evaluate_function_call_ast(
      "Rational",
      &[bigint_to_expr(in_n), bigint_to_expr(in_d)],
    )?
  };
  let sqrt =
    crate::evaluator::evaluate_function_call_ast("Sqrt", &[radicand_expr])?;
  crate::evaluator::evaluate_function_call_ast("Times", &[coeff_expr, sqrt])
}

/// SixJSymbol[{j1, j2, j3}, {j4, j5, j6}] via the Racah closed form:
///   6j = Δ(j1,j2,j3) Δ(j4,j5,j3) Δ(j4,j2,j6) Δ(j1,j5,j6) · Σ_t f(t)
/// where Δ(a,b,c) = √[(a+b-c)!(a-b+c)!(-a+b+c)!/(a+b+c+1)!] and the sum
/// over t covers the values for which every factorial argument is
/// non-negative. We accumulate the rational result symbolically as
/// `(rational coefficient) * Sqrt[rational under-radical]` and let the
/// evaluator combine these — for the all-integer triangle case this
/// collapses cleanly to a plain Rational.
pub fn six_j_symbol_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "SixJSymbol expects exactly 2 arguments".into(),
    ));
  }
  let unchanged = || unevaluated("SixJSymbol", args);
  let mut two_j = [0i128; 6];
  // Track exactly-one symbolic j-index for the Piecewise fallback.
  let mut symbolic: Option<(usize, String)> = None;
  for (group, arg) in args.iter().enumerate() {
    let Expr::List(items) = arg else {
      return Ok(unchanged());
    };
    if items.len() != 3 {
      return Ok(unchanged());
    }
    for (k, item) in items.iter().enumerate() {
      let pos = group * 3 + k;
      if let Some(v) = two_half_int(item) {
        if v < 0 {
          return Ok(unchanged());
        }
        two_j[pos] = v;
      } else if let Expr::Identifier(name) = item {
        if symbolic.is_some() {
          return Ok(unchanged());
        }
        symbolic = Some((pos, name.clone()));
        two_j[pos] = i128::MIN; // placeholder
      } else {
        return Ok(unchanged());
      }
    }
  }

  // Symbolic branch: try each non-negative integer j-value in 2j-units
  // and emit a Piecewise of (value, var == k) branches for the ones that
  // produce a non-zero coefficient.
  if let Some((idx, sym_name)) = symbolic {
    return six_j_symbol_symbolic(args, idx, sym_name, two_j);
  }
  let [j1, j2, j3, j4, j5, j6] = two_j;
  let triangles = [(j1, j2, j3), (j1, j5, j6), (j4, j2, j6), (j4, j5, j3)];
  for (a, b, c) in triangles {
    if c < (a - b).abs() || c > a + b || (a + b + c).rem_euclid(2) != 0 {
      crate::emit_message(&format!(
        "SixJSymbol::tri: SixJSymbol[{}, {}] is not triangular.",
        expr_to_string(&args[0]),
        expr_to_string(&args[1])
      ));
      return Ok(Expr::Integer(0));
    }
  }

  // Compute Δ²(a, b, c) = (a+b-c)!(a-b+c)!(-a+b+c)!/(a+b+c+1)! using
  // half-integer arithmetic — every triangle has an even sum so each of
  // (a±b±c)/2 is an integer.
  fn fact(n: i128) -> BigInt {
    let mut acc = BigInt::from(1);
    for i in 2..=n {
      acc *= i;
    }
    acc
  }
  fn delta_sq_num_den(a: i128, b: i128, c: i128) -> (BigInt, BigInt) {
    // Each triangle satisfies a+b+c even, so the half-integer factorials
    // collapse to integer factorials.
    let n1 = (a + b - c) / 2;
    let n2 = (a - b + c) / 2;
    let n3 = (-a + b + c) / 2;
    let nd = (a + b + c) / 2 + 1;
    (fact(n1) * fact(n2) * fact(n3), fact(nd))
  }
  // Δ_total² = Π Δ²(triangle_i). Accumulate as (num, den).
  let mut delta_sq_num = BigInt::from(1);
  let mut delta_sq_den = BigInt::from(1);
  for (a, b, c) in triangles {
    let (n, d) = delta_sq_num_den(a, b, c);
    delta_sq_num *= n;
    delta_sq_den *= d;
  }

  // Sum range for t (in units of 2j):
  //   t_min = max(j1+j2+j3, j1+j5+j6, j4+j2+j6, j4+j5+j3)
  //   t_max = min(j1+j2+j4+j5, j1+j3+j4+j6, j2+j3+j5+j6)
  let two_t_min = [j1 + j2 + j3, j1 + j5 + j6, j4 + j2 + j6, j4 + j5 + j3]
    .into_iter()
    .max()
    .unwrap();
  let two_t_max = [j1 + j2 + j4 + j5, j1 + j3 + j4 + j6, j2 + j3 + j5 + j6]
    .into_iter()
    .min()
    .unwrap();

  // The Racah sum runs over integer t = two_t / 2; since each two_t* is
  // a sum of (potentially half-)integers with consistent parity, both
  // bounds share that parity, so step by 2 in two-units.
  let mut sum_num = BigInt::from(0);
  let mut sum_den = BigInt::from(1);
  let mut two_t = two_t_min;
  while two_t <= two_t_max {
    if two_t.rem_euclid(2) != 0 {
      // Half-integer t — Racah's sum is over integer t; if either bound
      // is half-integer we'd be in a different branch. Skip just in case.
      two_t += 2;
      continue;
    }
    let t = two_t / 2;
    let term_num = fact(t + 1);
    // Denominators (each non-negative by construction).
    let denoms = [
      (two_t - (j1 + j2 + j3)) / 2,
      (two_t - (j1 + j5 + j6)) / 2,
      (two_t - (j4 + j2 + j6)) / 2,
      (two_t - (j4 + j5 + j3)) / 2,
      ((j1 + j2 + j4 + j5) - two_t) / 2,
      ((j1 + j3 + j4 + j6) - two_t) / 2,
      ((j2 + j3 + j5 + j6) - two_t) / 2,
    ];
    let mut term_den = BigInt::from(1);
    for d in denoms {
      term_den *= fact(d);
    }
    let sign = if t.rem_euclid(2) == 0 { 1 } else { -1 };
    // sum += sign * term_num / term_den
    //      = (sign * term_num * sum_den + sum_num * term_den)
    //        / (sum_den * term_den)
    let new_num =
      &sum_num * &term_den + BigInt::from(sign) * &term_num * &sum_den;
    let new_den = &sum_den * &term_den;
    sum_num = new_num;
    sum_den = new_den;
    two_t += 2;
  }

  // Result = sqrt(delta_sq_num / delta_sq_den) * (sum_num / sum_den).
  // Split delta_sq into a perfect-square scalar (pulled outside Sqrt) and
  // a square-free residue. Done with BigInt arithmetic so it scales past
  // the u64 trial-division Sqrt fall back to.
  let (delta_sq_outside_num, delta_sq_inside_num) =
    extract_perfect_square_bigint(&delta_sq_num);
  let (delta_sq_outside_den, delta_sq_inside_den) =
    extract_perfect_square_bigint(&delta_sq_den);
  // sqrt(delta_sq_num/delta_sq_den) = (out_n/out_d) * Sqrt[in_n/in_d]
  // result = (sum_num · out_n) / (sum_den · out_d) · Sqrt[in_n/in_d]
  let coeff_num = sum_num * &delta_sq_outside_num;
  let coeff_den = &sum_den * &delta_sq_outside_den;
  let coeff_expr = bigint_rational_to_expr(coeff_num, coeff_den);
  let radicand_expr = if delta_sq_inside_den == BigInt::from(1) {
    bigint_to_expr(delta_sq_inside_num)
  } else {
    crate::evaluator::evaluate_function_call_ast(
      "Rational",
      &[
        bigint_to_expr(delta_sq_inside_num),
        bigint_to_expr(delta_sq_inside_den),
      ],
    )?
  };
  // Evaluate Sqrt eagerly so a perfect-square residue collapses before
  // Times is applied (`evaluate_function_call_ast` does not descend into
  // args).
  let sqrt =
    crate::evaluator::evaluate_function_call_ast("Sqrt", &[radicand_expr])?;
  crate::evaluator::evaluate_function_call_ast("Times", &[coeff_expr, sqrt])
}

/// Symbolic-index branch for `SixJSymbol`: with exactly one of the six
/// j-arguments a bare `Identifier`, enumerate the non-negative integer
/// values for that position permitted by all four triangle inequalities
/// and emit a `Piecewise[{{value, var == k}, …}, 0]` keyed on each
/// supported `k`. Returns plain `0` when the support is empty.
fn six_j_symbol_symbolic(
  args: &[Expr],
  sym_pos: usize,
  sym_name: String,
  mut two_j: [i128; 6],
) -> Result<Expr, InterpreterError> {
  // The four triangle (a, b, c) tuples in terms of indices 0..6:
  //   (0, 1, 2), (0, 4, 5), (3, 1, 5), (3, 4, 2)
  let triangles: [(usize, usize, usize); 4] =
    [(0, 1, 2), (0, 4, 5), (3, 1, 5), (3, 4, 2)];

  // For each triangle that contains `sym_pos`, derive the constraint on
  // two_j[sym_pos]. A valid 2j must be a non-negative integer that
  // satisfies `|a - b| ≤ c ≤ a + b` plus `a + b + c ≡ 0 (mod 2)`. We
  // enforce the same `a + b + c` even parity across all four triangles
  // by collecting bounds for each triangle that contains sym_pos and
  // intersecting them.
  //
  // The sym 2j value `v` must be a non-negative integer in 2j-units;
  // we'll iterate over candidate values from 0 up to `j_max_total`
  // (the largest |a+b| from any containing triangle) and keep those
  // that pass every triangle check (both containing and not).
  let mut upper_bound: i128 = 0;
  for tri in &triangles {
    for &pos in &[tri.0, tri.1, tri.2] {
      if pos == sym_pos {
        let others: Vec<usize> = [tri.0, tri.1, tri.2]
          .iter()
          .copied()
          .filter(|&p| p != sym_pos)
          .collect();
        upper_bound = upper_bound.max(two_j[others[0]] + two_j[others[1]]);
      }
    }
  }
  // Triangles not containing sym_pos must already be valid; if any
  // fails, the whole symbol is 0.
  for &(a, b, c) in &triangles {
    if a != sym_pos && b != sym_pos && c != sym_pos {
      let ta = two_j[a];
      let tb = two_j[b];
      let tc = two_j[c];
      if tc < (ta - tb).abs()
        || tc > ta + tb
        || (ta + tb + tc).rem_euclid(2) != 0
      {
        return Ok(Expr::Integer(0));
      }
    }
  }

  // Enumerate candidate symbolic j-values in 2j-units. We only emit
  // branches for integer (not half-integer) j — wolframscript's
  // Piecewise condition restricts to integer m in our audit case.
  let mut branches: Vec<Expr> = Vec::new();
  for v_2j in (0..=upper_bound).step_by(2) {
    two_j[sym_pos] = v_2j;
    // Validate all triangles for this candidate.
    let mut ok = true;
    for &(a, b, c) in &triangles {
      let ta = two_j[a];
      let tb = two_j[b];
      let tc = two_j[c];
      if tc < (ta - tb).abs()
        || tc > ta + tb
        || (ta + tb + tc).rem_euclid(2) != 0
      {
        ok = false;
        break;
      }
    }
    if !ok {
      continue;
    }
    // Compute the concrete SixJSymbol by rebuilding args with the
    // forced value substituted in.
    let group = sym_pos / 3;
    let k = sym_pos % 3;
    let mut new_args = args.to_vec();
    if let Expr::List(items) = &new_args[group] {
      let mut new_items: Vec<Expr> = items.to_vec();
      let v_j = v_2j / 2;
      let forced = if v_2j.rem_euclid(2) == 0 {
        Expr::Integer(v_j)
      } else {
        crate::functions::math_ast::make_rational(v_2j, 2)
      };
      new_items[k] = forced.clone();
      new_args[group] = Expr::List(new_items.into());
    }
    let value = six_j_symbol_ast(&new_args)?;
    if matches!(&value, Expr::Integer(0)) {
      continue;
    }
    let forced_value = Expr::Integer(v_2j / 2);
    let cond = Expr::Comparison {
      operands: vec![Expr::Identifier(sym_name.clone()), forced_value],
      operators: vec![ComparisonOp::Equal],
    };
    branches.push(Expr::List(vec![value, cond].into()));
  }

  if branches.is_empty() {
    return Ok(Expr::Integer(0));
  }
  Ok(Expr::FunctionCall {
    name: "Piecewise".to_string(),
    args: vec![Expr::List(branches.into()), Expr::Integer(0)].into(),
  })
}

/// Split `n` (assumed positive) into `(outside, inside)` such that
/// `n == outside^2 · inside` and `inside` is square-free. Operates on
/// BigInt directly so it works for arguments far beyond u64 range — the
/// general `Sqrt` path trial-divides over u64.
fn extract_perfect_square_bigint(n: &BigInt) -> (BigInt, BigInt) {
  use num_traits::Zero;
  if n.is_zero() {
    return (BigInt::from(1), BigInt::from(0));
  }
  let mut outside = BigInt::from(1);
  let mut inside = n.clone();
  let mut factor = BigInt::from(2);
  loop {
    let fsq = &factor * &factor;
    if fsq > inside {
      break;
    }
    while (&inside % &fsq).is_zero() {
      outside *= &factor;
      inside /= &fsq;
    }
    factor += 1;
  }
  (outside, inside)
}

fn bigint_rational_to_expr(num: BigInt, den: BigInt) -> Expr {
  use num_traits::Zero;
  if den.is_zero() {
    return Expr::Identifier("ComplexInfinity".to_string());
  }
  let g = gcd_bigint(&num, &den);
  let (mut n, mut d) = (num / &g, den / &g);
  if d < BigInt::from(0) {
    n = -n;
    d = -d;
  }
  if d == BigInt::from(1) {
    return bigint_to_expr(n);
  }
  Expr::FunctionCall {
    name: "Rational".to_string(),
    args: vec![bigint_to_expr(n), bigint_to_expr(d)].into(),
  }
}

/// Like `two_half_int` but preserves sign — used for the m projections in
/// ThreeJSymbol, which may be negative.
fn two_half_int_signed(expr: &Expr) -> Option<i128> {
  // UnaryOp::Minus wrapping an integer/rational.
  if let Expr::UnaryOp {
    op: UnaryOperator::Minus,
    operand,
  } = expr
  {
    return two_half_int_signed(operand).map(|v| -v);
  }
  match expr_to_rational(expr) {
    Some((n, 1)) => n.checked_mul(2),
    Some((n, 2)) => Some(n),
    _ => None,
  }
}

/// ClebschGordan[{j1, m1}, {j2, m2}, {j, m}]
///
/// Currently implements only the cases that evaluate to easily-determined
/// values:
/// - Returns 0 for degenerate inputs (invalid projections, triangle failure,
///   integer-parity mismatch).
/// - Returns 1 for the "stretched state" m1 = ±j1, m2 = ±j2 (same sign),
///   m = m1 + m2 = ±(j1+j2), j = j1+j2.
///
/// All other (valid) cases return the call unchanged.
pub fn clebsch_gordan_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "ClebschGordan expects exactly 3 arguments".into(),
    ));
  }
  let unchanged = || unevaluated("ClebschGordan", args);
  let mut two_j = [0i128; 3];
  let mut two_m = [0i128; 3];
  // Track which `m_i` (if any) is symbolic so we can fall through to the
  // Piecewise path below.
  let mut symbolic_m: Option<(usize, String)> = None;
  for (i, arg) in args.iter().enumerate() {
    let Expr::List(items) = arg else {
      return Ok(unchanged());
    };
    if items.len() != 2 {
      return Ok(unchanged());
    }
    let Some(tj) = two_half_int(&items[0]) else {
      return Ok(unchanged());
    };
    if tj < 0 {
      return Ok(unchanged());
    }
    two_j[i] = tj;
    if let Some(tm) = two_half_int_signed(&items[1]) {
      two_m[i] = tm;
    } else if let Expr::Identifier(name) = &items[1] {
      // Symbolic m_i — only allow exactly one.
      if symbolic_m.is_some() {
        return Ok(unchanged());
      }
      symbolic_m = Some((i, name.clone()));
      // Placeholder until we solve the constraint below.
      two_m[i] = i128::MIN;
    } else {
      return Ok(unchanged());
    }
  }

  // Symbolic-m branch: solve m1 + m2 = m3 for the missing projection,
  // then emit a Piecewise wrapping the concrete CG at that value.
  if let Some((idx, sym_name)) = symbolic_m {
    // Solve for the missing two_m[idx].
    two_m[idx] = match idx {
      0 => two_m[2] - two_m[1], // m1 = m3 - m2
      1 => two_m[2] - two_m[0], // m2 = m3 - m1
      2 => two_m[0] + two_m[1], // m3 = m1 + m2
      _ => unreachable!(),
    };
    // The forced m_idx is in 2m-units; the externally-visible m is
    // half of that. If 2m is odd we need a half-integer Rational; the
    // condition itself stays `m == forced` regardless.
    let forced_m_expr = if two_m[idx].rem_euclid(2) == 0 {
      Expr::Integer(two_m[idx] / 2)
    } else {
      crate::functions::math_ast::make_rational(two_m[idx], 2)
    };
    // Build the concrete-m args and recurse so all the existing
    // selection-rule machinery runs.
    let mut concrete_args: Vec<Expr> = args.to_vec();
    if let Expr::List(items) = &concrete_args[idx]
      && items.len() == 2
    {
      let mut new_items: Vec<Expr> = items.to_vec();
      new_items[1] = forced_m_expr.clone();
      concrete_args[idx] = Expr::List(new_items.into());
    }
    let value = clebsch_gordan_ast(&concrete_args)?;
    // If the concrete CG evaluates to a literal 0 (i.e. the forced m
    // violates |m_i| ≤ j_i or some other rule), the Piecewise has no
    // non-zero branch — just return 0.
    if matches!(&value, Expr::Integer(0)) {
      return Ok(Expr::Integer(0));
    }
    let condition = Expr::Comparison {
      operands: vec![Expr::Identifier(sym_name.clone()), forced_m_expr],
      operators: vec![ComparisonOp::Equal],
    };
    let branch = Expr::List(vec![value, condition].into());
    let pw_args = Expr::List(vec![branch].into());
    return Ok(Expr::FunctionCall {
      name: "Piecewise".to_string(),
      args: vec![pw_args, Expr::Integer(0)].into(),
    });
  }

  let (j1, j2, j3) = (two_j[0], two_j[1], two_j[2]);
  let (m1, m2, m3) = (two_m[0], two_m[1], two_m[2]);

  // m1 + m2 must equal m.
  if m1 + m2 != m3 {
    return Ok(Expr::Integer(0));
  }
  for i in 0..3 {
    if two_m[i].abs() > two_j[i] {
      return Ok(Expr::Integer(0));
    }
    // j_i - m_i must be an integer, i.e. 2j_i and 2m_i share parity.
    if (two_j[i] - two_m[i]).rem_euclid(2) != 0 {
      return Ok(Expr::Integer(0));
    }
  }
  // Triangle inequality on j1, j2, j3.
  if j3 < (j1 - j2).abs() || j3 > j1 + j2 {
    return Ok(Expr::Integer(0));
  }
  if (j1 + j2 + j3).rem_euclid(2) != 0 {
    return Ok(Expr::Integer(0));
  }

  // Stretched state: j = j1 + j2, |m_i| = j_i with matching signs.
  if j3 == j1 + j2
    && m1.abs() == j1
    && m2.abs() == j2
    && m1.signum() * m2.signum() >= 0
    && m1 + m2 == m3
  {
    return Ok(Expr::Integer(1));
  }

  // General case via the standard CG↔3j relation:
  //   CG[{j1,m1},{j2,m2},{j,m}] = (-1)^(j1-j2+m) · Sqrt[2j+1]
  //                                · ThreeJSymbol[{j1,m1},{j2,m2},{j,-m}]
  let neg_m3 = bigint_to_expr(BigInt::from(-m3)).pipe_through_rational(2);
  let three_j_args = vec![
    Expr::List(
      vec![
        bigint_to_expr(BigInt::from(j1)).pipe_through_rational(2),
        bigint_to_expr(BigInt::from(m1)).pipe_through_rational(2),
      ]
      .into(),
    ),
    Expr::List(
      vec![
        bigint_to_expr(BigInt::from(j2)).pipe_through_rational(2),
        bigint_to_expr(BigInt::from(m2)).pipe_through_rational(2),
      ]
      .into(),
    ),
    Expr::List(
      vec![
        bigint_to_expr(BigInt::from(j3)).pipe_through_rational(2),
        neg_m3,
      ]
      .into(),
    ),
  ];
  let three_j = three_j_symbol_ast(&three_j_args)?;
  // (-1)^(j1 - j2 + m3): the parity check above guarantees this exponent
  // is an integer.
  let outer_sign_exp = (j1 - j2 + m3) / 2;
  let outer_sign: i128 = if outer_sign_exp.rem_euclid(2) == 0 {
    1
  } else {
    -1
  };
  // Sqrt[2j3 + 1] in 2j-units: 2j3 + 1 = j3 + 1 because j3 here is the
  // _doubled_ value. Wait — j3 above is `two_j[2]`, so the actual angular
  // momentum is j3/2 and 2(j3/2) + 1 = j3 + 1. ✓
  let sqrt_factor = crate::evaluator::evaluate_function_call_ast(
    "Sqrt",
    &[Expr::Integer(j3 + 1)],
  )?;
  crate::evaluator::evaluate_function_call_ast(
    "Times",
    &[Expr::Integer(outer_sign), sqrt_factor, three_j],
  )
}

/// Helper for `clebsch_gordan_ast`: convert an integer in 2j (or 2m)
/// units back to a half-integer Expr (i.e., divide by `denom`).
trait PipeThroughRational {
  fn pipe_through_rational(self, denom: i128) -> Expr;
}
impl PipeThroughRational for Expr {
  fn pipe_through_rational(self, denom: i128) -> Expr {
    if denom == 1 {
      return self;
    }
    if let Expr::Integer(n) = self {
      if n.rem_euclid(denom) == 0 {
        return Expr::Integer(n / denom);
      }
      return crate::functions::math_ast::make_rational(n, denom);
    }
    self
  }
}

/// FareySequence[n] - ascending Farey fractions of order n;
/// FareySequence[n, k] - the k-th element (1-indexed).
///
/// Message policy matches wolframscript: a non-positive integer order
/// emits FareySequence::intpm and yields Null in the 1-argument form
/// only; an out-of-range positive integer rank emits
/// FareySequence::rank and stays unevaluated; everything else
/// (symbolic, rational, rank 0) stays silently unevaluated.
pub fn farey_sequence_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated("FareySequence", args);
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated(args));
  }
  let n = match &args[0] {
    Expr::Integer(k) if *k >= 1 => *k,
    Expr::Integer(_) if args.len() == 1 => {
      crate::emit_message(&format!(
        "FareySequence::intpm: Positive machine-sized integer expected at position 1 in {}.",
        expr_to_string(&unevaluated(args))
      ));
      return Ok(Expr::Identifier("Null".to_string()));
    }
    _ => return Ok(unevaluated(args)),
  };

  // Farey neighbor recurrence: from adjacent a/b, c/d the next term is
  // (k*c - a)/(k*d - b) with k = (n + b) div d.
  let mut terms: Vec<Expr> = Vec::new();
  let (mut a, mut b, mut c, mut d) = (0i128, 1i128, 1i128, n);
  terms.push(Expr::Integer(0));
  while c <= n {
    let k = (n + b) / d;
    let (c2, d2) = (k * c - a, k * d - b);
    a = c;
    b = d;
    c = c2;
    d = d2;
    terms.push(make_rational_expr(BigInt::from(a), BigInt::from(b)));
  }

  if args.len() == 1 {
    return Ok(Expr::List(terms.into()));
  }
  match &args[1] {
    Expr::Integer(rank) if *rank >= 1 => {
      if *rank as usize > terms.len() {
        crate::emit_message(&format!(
          "FareySequence::rank: Farey sequence rank {} at position 2 of {} is expected to be a positive integer less than or equal to {}.",
          rank,
          expr_to_string(&unevaluated(args)),
          terms.len()
        ));
        return Ok(unevaluated(args));
      }
      Ok(terms.swap_remove(*rank as usize - 1))
    }
    _ => Ok(unevaluated(args)),
  }
}

/// Fibonorial[n] - product of the first n Fibonacci numbers.
/// Negative integers give ComplexInfinity; non-integer numbers emit
/// Fibonorial::intnm and stay unevaluated; symbols stay quiet.
pub fn fibonorial_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated("Fibonorial", args);
  if args.len() != 1 {
    return Ok(unevaluated(args));
  }
  let is_non_integer_number = match &args[0] {
    Expr::Real(_) => true,
    Expr::FunctionCall { name, .. } if name == "Rational" => true,
    _ => false,
  };
  match &args[0] {
    Expr::Integer(n) if *n >= 0 => {
      let mut product = BigInt::from(1);
      let (mut a, mut b) = (BigInt::from(1), BigInt::from(1));
      for _ in 0..*n {
        product *= &a;
        let next = &a + &b;
        a = b;
        b = next;
      }
      Ok(crate::functions::math_ast::numeric_utils::bigint_to_expr(
        product,
      ))
    }
    Expr::Integer(_) => Ok(Expr::Identifier("ComplexInfinity".to_string())),
    _ if is_non_integer_number => {
      crate::emit_message(&format!(
        "Fibonorial::intnm: Non-negative machine-sized integer expected at position 1 in {}.",
        expr_to_string(&unevaluated(args))
      ));
      Ok(unevaluated(args))
    }
    _ => Ok(unevaluated(args)),
  }
}
