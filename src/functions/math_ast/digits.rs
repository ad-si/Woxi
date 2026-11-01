use super::*;
use num_traits::{Signed, ToPrimitive, Zero};

/// DigitCount[n] - counts of each digit 1-9,0 in base 10
/// DigitCount[n, b] - counts of each digit in base b
/// DigitCount[n, b, d] - count of specific digit d in base b
pub fn digit_count_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 4 {
    return Err(InterpreterError::EvaluationError(
      "DigitCount expects 1 to 4 arguments".into(),
    ));
  }
  let n = match expr_to_bigint(&args[0]) {
    Some(n) => n.abs(),
    None => {
      return Ok(unevaluated("DigitCount", args));
    }
  };
  let base = if args.len() >= 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => b,
      _ => {
        return Ok(unevaluated("DigitCount", args));
      }
    }
  } else {
    10
  };

  // Get digit list in the given base
  let big_base = BigInt::from(base);
  let mut digits = Vec::new();
  let mut val = n;
  if val.is_zero() {
    digits.push(0usize);
  } else {
    while !val.is_zero() {
      use num_traits::ToPrimitive;
      let rem = (&val % &big_base).to_usize().unwrap_or(0);
      digits.push(rem);
      val /= &big_base;
    }
  }

  // DigitCount[n, b, d, len]: only the last len digits, padded on the left
  // with zeros when n has fewer.
  if let Some(len) = args.get(3) {
    match expr_to_i128(len) {
      Some(len) if len >= 0 => digits.resize(len as usize, 0),
      _ => return Ok(unevaluated("DigitCount", args)),
    }
  }

  if args.len() >= 3 {
    // DigitCount[n, b, d] - count of specific digit d; a list of digits
    // threads.
    let count_of = |d: &Expr| -> Option<Expr> {
      let d = expr_to_i128(d)?;
      Some(Expr::Integer(
        digits.iter().filter(|&&x| x as i128 == d).count() as i128,
      ))
    };
    let result = match &args[2] {
      Expr::List(ds) => ds
        .iter()
        .map(count_of)
        .collect::<Option<Vec<_>>>()
        .map(|v| Expr::List(v.into())),
      d => count_of(d),
    };
    Ok(result.unwrap_or_else(|| unevaluated("DigitCount", args)))
  } else {
    // DigitCount[n] or DigitCount[n, b] - list of counts for digits 1..base-1, 0
    // Wolfram returns counts in order: digit 1, digit 2, ..., digit (base-1), digit 0
    let mut counts = vec![0i128; base as usize];
    for &d in &digits {
      counts[d] += 1;
    }
    // Reorder: digits 1, 2, ..., base-1, 0
    let mut result = Vec::with_capacity(base as usize);
    for d in 1..base as usize {
      result.push(Expr::Integer(counts[d]));
    }
    result.push(Expr::Integer(counts[0]));
    Ok(Expr::List(result.into()))
  }
}

/// DigitSum[n] - sum of digits in base 10
/// DigitSum[n, b] - sum of digits in base b
/// DigitSum[n, b, k] - sum of the k-th powers of the base-b digits
pub fn digit_sum_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "DigitSum expects 1 to 3 arguments".into(),
    ));
  }
  // The power each digit is raised to (1 unless a third argument is given).
  let power: u32 = match args.get(2) {
    None => 1,
    Some(Expr::Integer(k)) if *k >= 0 && *k <= u32::MAX as i128 => *k as u32,
    Some(_) => return Ok(unevaluated("DigitSum", args)),
  };
  let add_digit = |sum: &mut BigInt, d: BigInt| {
    *sum += if power == 1 { d } else { d.pow(power) };
  };
  let n = match expr_to_bigint(&args[0]) {
    Some(n) => n.abs(),
    None => {
      return Ok(unevaluated("DigitSum", args));
    }
  };
  // DigitSum[n, MixedRadix[{b1, b2, ..., bk}]]
  if args.len() == 2
    && let Expr::FunctionCall {
      name: mname,
      args: margs,
    } = &args[1]
    && mname == "MixedRadix"
    && margs.len() == 1
    && let Expr::List(bases) = &margs[0]
  {
    let mut base_vals: Vec<BigInt> = Vec::with_capacity(bases.len());
    for b in bases {
      let Some(bi) = expr_to_i128(b) else {
        return Ok(unevaluated("DigitSum", args));
      };
      // A MixedRadix specification may legitimately contain a radix of 1
      // (e.g. MixedRadix[{60, 60, 1}]); only reject non-positive radices.
      if bi < 1 {
        return Ok(unevaluated("DigitSum", args));
      }
      base_vals.push(BigInt::from(bi));
    }
    let mut sum = BigInt::from(0);
    let mut val = n;
    // Right-to-left: rightmost digit uses last base, etc.
    for b in base_vals.iter().rev() {
      if val.is_zero() {
        break;
      }
      add_digit(&mut sum, &val % b);
      val /= b;
    }
    // Any remaining value is the overflow digit (no base constraint).
    if !val.is_zero() {
      add_digit(&mut sum, val);
    }
    return Ok(bigint_to_expr(sum));
  }

  let base = if args.len() >= 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => b,
      _ => {
        return Ok(unevaluated("DigitSum", args));
      }
    }
  } else {
    10
  };
  let big_base = BigInt::from(base);
  let mut sum = BigInt::from(0);
  let mut val = n;
  if val.is_zero() {
    return Ok(Expr::Integer(0));
  }
  while !val.is_zero() {
    add_digit(&mut sum, &val % &big_base);
    val /= &big_base;
  }
  Ok(bigint_to_expr(sum))
}

// --- High-precision constants for ContinuedFraction ---

/// Decimal digits of working precision for the first `n` continued-fraction
/// terms of `name`. The expansion is only trusted while the truncation error
/// stays below `1/q_n²`, where `q_n` is the denominator of the n-th
/// convergent. For a generic irrational like π a term costs about 0.515
/// digits of `log10 q_n`, so `1.1 n` digits leave a margin. The terms of `e`
/// grow (`1, 2k, 1, …`), so `log10 q_n` is about `(n/3)·log10(2n/3)` and the
/// precision has to grow superlinearly. The constant offset covers small `n`.
fn cf_working_digits(name: &str, n: usize) -> usize {
  let n_f = n as f64;
  let digits = if name == "E" {
    let third = (n_f / 3.0).max(1.0);
    2.0 * third * (2.0 * third).max(10.0).log10()
  } else {
    1.1 * n_f
  };
  digits.ceil() as usize + 40
}

/// `atan(1/k) · scale`, truncated, from the Taylor series
/// `1/k − 1/(3k³) + 1/(5k⁵) − …`.
fn atan_recip_scaled(k: u32, scale: &num_bigint::BigInt) -> num_bigint::BigInt {
  use num_bigint::BigInt;
  let k2 = BigInt::from(k as u64 * k as u64);
  let mut power = scale / BigInt::from(k);
  let mut sum = power.clone();
  let mut n: u64 = 1;
  loop {
    power /= &k2;
    if power.is_zero() {
      return sum;
    }
    let term = &power / BigInt::from(2 * n + 1);
    if n % 2 == 1 {
      sum -= term;
    } else {
      sum += term;
    }
    n += 1;
  }
}

/// `π · scale` from Machin's formula `π = 16·atan(1/5) − 4·atan(1/239)`.
fn pi_scaled(scale: &num_bigint::BigInt) -> num_bigint::BigInt {
  atan_recip_scaled(5, scale) * 16 - atan_recip_scaled(239, scale) * 4
}

/// `e · scale` from the series `Σ 1/k!`.
fn e_scaled(scale: &num_bigint::BigInt) -> num_bigint::BigInt {
  use num_bigint::BigInt;
  let mut term = scale.clone();
  let mut sum = scale.clone();
  let mut k: u64 = 1;
  loop {
    term /= BigInt::from(k);
    if term.is_zero() {
      return sum;
    }
    sum += &term;
    k += 1;
  }
}

/// First `n` continued-fraction terms of `numerator / denominator`.
fn continued_fraction_of_ratio(
  numerator: &num_bigint::BigInt,
  denominator: num_bigint::BigUint,
  n: usize,
) -> Vec<i128> {
  use num_bigint::{BigInt, Sign};
  use num_traits::{One, ToPrimitive};
  let mut result = Vec::new();
  if n == 0 {
    return result;
  }
  // The first term is a floor, which rounds away from zero for a negative
  // value; every later remainder is positive.
  let negative = numerator.sign() == Sign::Minus;
  let magnitude = numerator.magnitude();
  let (q, r) = (magnitude / &denominator, magnitude % &denominator);
  let (mut a, mut b) = (denominator, r);
  let first: BigInt = if negative {
    if b.is_zero() {
      -BigInt::from(q)
    } else {
      b = &a - &b;
      -BigInt::from(q) - BigInt::one()
    }
  } else {
    BigInt::from(q)
  };
  result.push(first.to_i128().unwrap_or(0));
  while result.len() < n && !b.is_zero() {
    // current value is a / b with a > b
    let (q, r) = (&a / &b, &a % &b);
    result.push(q.to_i128().unwrap_or(0));
    (a, b) = (b, r);
  }
  result
}

/// Try to compute a constant expression's first `n_terms` continued-fraction
/// terms. Returns None if the expression is not a recognized constant.
fn try_constant_continued_fraction(
  expr: &Expr,
  n_terms: usize,
) -> Option<Vec<i128>> {
  use num_bigint::BigInt;
  let (name, negative) = match expr {
    Expr::Constant(name) => (name.as_str(), false),
    _ => return None,
  };
  let (name, negative) = match name.strip_prefix('-') {
    Some(rest) => (rest, !negative),
    None => (name, negative),
  };
  let digits = cf_working_digits(name, n_terms);
  let scale = BigInt::from(10u32).pow(digits as u32);
  let scaled = match name {
    "Pi" => pi_scaled(&scale),
    "E" => e_scaled(&scale),
    _ => return None,
  };
  let numerator = if negative { -scaled } else { scaled };
  Some(continued_fraction_of_ratio(
    &numerator,
    scale.magnitude().clone(),
    n_terms,
  ))
}

/// Extract the integer D from a Sqrt[D] expression. Handles both the
/// `Sqrt[D]` FunctionCall shape and the canonicalized
/// `Power[D, Rational[1, 2]]` shape (BinaryOp or FunctionCall).
fn extract_sqrt_integer(expr: &Expr) -> Option<i128> {
  match expr {
    Expr::FunctionCall { name, args } if name == "Sqrt" && args.len() == 1 => {
      if let Expr::Integer(d) = &args[0] {
        return Some(*d);
      }
      None
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if let (Expr::Integer(d), Expr::FunctionCall { name: rn, args: ra }) =
        (&args[0], &args[1])
        && rn == "Rational"
        && ra.len() == 2
        && matches!(ra[0], Expr::Integer(1))
        && matches!(ra[1], Expr::Integer(2))
      {
        return Some(*d);
      }
      None
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      if let (Expr::Integer(d), Expr::FunctionCall { name: rn, args: ra }) =
        (left.as_ref(), right.as_ref())
        && rn == "Rational"
        && ra.len() == 2
        && matches!(ra[0], Expr::Integer(1))
        && matches!(ra[1], Expr::Integer(2))
      {
        return Some(*d);
      }
      None
    }
    _ => None,
  }
}

/// Match 1/√d (Power[d, Rational[-1, 2]], the canonical form of
/// Sqrt[d]/d) and return d.
fn extract_reciprocal_sqrt_integer(expr: &Expr) -> Option<i128> {
  let rational_minus_half = |e: &Expr| -> bool {
    matches!(e, Expr::FunctionCall { name, args }
      if name == "Rational"
        && args.len() == 2
        && matches!(args[0], Expr::Integer(-1))
        && matches!(args[1], Expr::Integer(2)))
  };
  match expr {
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if let Expr::Integer(d) = &args[0]
        && rational_minus_half(&args[1])
      {
        return Some(*d);
      }
      None
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      if let Expr::Integer(d) = left.as_ref()
        && rational_minus_half(right)
      {
        return Some(*d);
      }
      None
    }
    _ => None,
  }
}

/// Extract a quadratic-irrational expression of the form `(p + q · √d) / r`
/// where `p`, `q`, `r` are integers and `d` is a positive integer (not a
/// perfect square — caller verifies). Returns `None` if the input doesn't
/// match.
fn extract_quadratic_irrational(
  expr: &Expr,
) -> Option<(i128, i128, i128, i128)> {
  // 1/√d = √d/d = (0 + 1·√d)/d
  if let Some(d) = extract_reciprocal_sqrt_integer(expr) {
    return Some((0, 1, d, d));
  }
  // Peel off outer scale: (scale_num/scale_den) · inner, collecting the
  // overall Rational coefficient and whatever Plus-sum remains.
  let (inner_owned, scale_num, scale_den): (Expr, i128, i128) = match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      if let Expr::Integer(r) = right.as_ref() {
        (left.as_ref().clone(), 1, *r)
      } else {
        (expr.clone(), 1, 1)
      }
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut scale_n: i128 = 1;
      let mut scale_d: i128 = 1;
      let mut remaining: Vec<Expr> = Vec::new();
      for a in args {
        match a {
          Expr::Integer(n) => scale_n = scale_n.checked_mul(*n)?,
          Expr::FunctionCall { name, args: rargs }
            if name == "Rational" && rargs.len() == 2 =>
          {
            if let (Expr::Integer(n), Expr::Integer(d)) = (&rargs[0], &rargs[1])
            {
              scale_n = scale_n.checked_mul(*n)?;
              scale_d = scale_d.checked_mul(*d)?;
            } else {
              return None;
            }
          }
          _ => remaining.push(a.clone()),
        }
      }
      let inner = match remaining.len() {
        0 => return None,
        1 => remaining.into_iter().next().unwrap(),
        _ => times(remaining),
      };
      (inner, scale_n, scale_d)
    }
    _ => (expr.clone(), 1, 1),
  };
  let inner = &inner_owned;
  // Inner: a sum of rationals and rational multiples of one Sqrt[d], as
  // the exact fractions p = pn/pd and q = qn/qd.
  let rational = |e: &Expr| -> Option<(i128, i128)> {
    match e {
      Expr::Integer(n) => Some((*n, 1)),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        match (&args[0], &args[1]) {
          (Expr::Integer(n), Expr::Integer(d)) => Some((*n, *d)),
          _ => None,
        }
      }
      _ => None,
    }
  };
  let add =
    |(an, ad): (i128, i128), (bn, bd): (i128, i128)| -> Option<(i128, i128)> {
      let n = an.checked_mul(bd)?.checked_add(bn.checked_mul(ad)?)?;
      let d = ad.checked_mul(bd)?;
      let g = gcd_i128(n, d).max(1);
      Some((n / g, d / g))
    };
  let (mut p, mut q): ((i128, i128), (i128, i128)) = ((0, 1), (0, 1));
  let mut d: i128 = 0;
  let terms: Vec<&Expr> = match inner {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => vec![left.as_ref(), right.as_ref()],
    Expr::FunctionCall { name, args } if name == "Plus" => {
      args.iter().collect()
    }
    _ => vec![inner],
  };
  for t in &terms {
    if let Some(r) = rational(t) {
      p = add(p, r)?;
      continue;
    }
    // Sqrt[d], or k * Sqrt[d] (canonically Times[k, Sqrt[d]]).
    let (coeff, sqrt_expr) = match t {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => (rational(left)?, right.as_ref()),
      Expr::FunctionCall { name, args }
        if name == "Times" && args.len() == 2 =>
      {
        (rational(&args[0])?, &args[1])
      }
      _ => ((1, 1), *t),
    };
    let dd = extract_sqrt_integer(sqrt_expr)?;
    q = add(q, coeff)?;
    if d != 0 && d != dd {
      return None;
    }
    d = dd;
  }
  if q.0 == 0 || d == 0 {
    return None;
  }
  // Compose with the outer scale: (scale_num/scale_den) · (p + q√d)
  //   = (sn·pn·qd + sn·qn·pd·√d) / (sd·pd·qd).
  let final_p = scale_num.checked_mul(p.0)?.checked_mul(q.1)?;
  let final_q = scale_num.checked_mul(q.0)?.checked_mul(p.1)?;
  let final_r = scale_den.checked_mul(p.1)?.checked_mul(q.1)?;
  let g = gcd_i128(gcd_i128(final_p, final_q), final_r).max(1);
  Some((final_p / g, final_q / g, d, final_r / g))
}

fn is_perfect_square(d: i128) -> bool {
  if d < 0 {
    return false;
  }
  let s = integer_sqrt(d);
  s * s == d
}

/// Continued-fraction expansion of a quadratic irrational `(p + q·√d) / r`
/// where `d` is a positive non-square integer. Returns
/// `Some((non_periodic_prefix, period))`.
fn continued_fraction_of_quadratic(
  p: i128,
  q: i128,
  d: i128,
  r: i128,
) -> Option<(Vec<i128>, Vec<i128>)> {
  if q == 0 || r == 0 {
    return None;
  }
  // A negative value expands as its negation does, every term negated:
  // ContinuedFraction[-Sqrt[2]] is {-1, {-2}}.
  let value = (p as f64 + q as f64 * (d as f64).sqrt()) / r as f64;
  if value < 0.0 {
    let (pre, period) = continued_fraction_of_quadratic(-p, -q, d, r)?;
    let negate = |v: Vec<i128>| v.into_iter().map(|t| -t).collect();
    return Some((negate(pre), negate(period)));
  }
  // Normalize to the form `(P + √N) / Q` with `Q | (N − P²)`: the root's
  // sign moves into P and Q, and while Q does not divide N − P², P and Q
  // are multiplied by |Q| (and N by Q²).
  let (mut big_p, mut big_q) = if q > 0 { (p, r) } else { (-p, -r) };
  let mut big_n: i128 = q.checked_mul(q)?.checked_mul(d)?;
  for _ in 0..4 {
    let diff = big_n.checked_sub(big_p.checked_mul(big_p)?)?;
    if diff.rem_euclid(big_q) == 0 {
      break;
    }
    let scale = big_q.abs();
    big_p = big_p.checked_mul(scale)?;
    big_n = big_n.checked_mul(scale)?.checked_mul(scale)?;
    big_q = big_q.checked_mul(scale)?;
  }
  if big_n
    .checked_sub(big_p.checked_mul(big_p)?)?
    .rem_euclid(big_q)
    != 0
  {
    return None;
  }

  // Iterate: a_k = floor((P + √N) / Q); P' = a·Q − P; Q' = (N − P'²) / Q.
  // With s = ⌊√N⌋ < √N < s + 1, the floor is ⌊(P + s)/Q⌋ for Q > 0 and
  // ⌊(P + s + 1)/Q⌋ for Q < 0.
  let sqrt_n_floor = integer_sqrt(big_n);
  let mut seen: std::collections::HashMap<(i128, i128), usize> =
    std::collections::HashMap::new();
  let mut terms: Vec<i128> = Vec::new();
  let mut p_k = big_p;
  let mut q_k = big_q;
  for step in 0..1000 {
    if let Some(&start) = seen.get(&(p_k, q_k)) {
      // wolframscript always keeps at least one term in the non-periodic
      // prefix (ContinuedFraction[GoldenRatio] == {1, {1}}, not {{1}}).
      // When the CF is purely periodic, pull the first period term into the
      // prefix and keep a copy at the start of the period — it's the same
      // value, just presented as "prefix + cycle".
      if start == 0 && !terms.is_empty() {
        let pre = vec![terms[0]];
        let mut period: Vec<i128> = terms[1..].to_vec();
        period.push(terms[0]);
        return Some((pre, period));
      }
      let pre = terms[..start].to_vec();
      let period = terms[start..].to_vec();
      return Some((pre, period));
    }
    seen.insert((p_k, q_k), step);
    let num = if q_k > 0 {
      p_k + sqrt_n_floor
    } else {
      p_k + sqrt_n_floor + 1
    };
    let a =
      num.div_euclid(q_k) - i128::from(num.rem_euclid(q_k) != 0 && q_k < 0);
    terms.push(a);
    let next_p = a.checked_mul(q_k)?.checked_sub(p_k)?;
    let new_q = big_n.checked_sub(next_p.checked_mul(next_p)?)? / q_k;
    if new_q == 0 {
      return None;
    }
    p_k = next_p;
    q_k = new_q;
  }
  None
}

/// Continued-fraction expansion of √D for a positive non-square integer D.
/// Returns `Some((a0, period))` where period is empty when D is a perfect
/// square. Returns None if D is non-positive.
fn continued_fraction_of_sqrt(d: i128) -> Option<(i128, Vec<i128>)> {
  if d <= 0 {
    return None;
  }
  // a0 = floor(sqrt(D))
  let a0 = integer_sqrt(d);
  if a0 * a0 == d {
    return Some((a0, Vec::new()));
  }

  // Standard algorithm: m_{n+1} = d_n·a_n − m_n;
  //                    d_{n+1} = (D − m_{n+1}²) / d_n;
  //                    a_{n+1} = floor((a_0 + m_{n+1}) / d_{n+1}).
  // The period ends when (m, d) first repeats — equivalently when
  // a_n == 2·a_0 (a known property of √D continued fractions).
  let mut period: Vec<i128> = Vec::new();
  let mut m: i128 = 0;
  let mut den: i128 = 1;
  let mut a: i128 = a0;
  for _ in 0..10_000 {
    let m_next = den * a - m;
    let d_next = (d - m_next * m_next) / den;
    if d_next == 0 {
      return None;
    }
    let a_next = (a0 + m_next) / d_next;
    period.push(a_next);
    m = m_next;
    den = d_next;
    a = a_next;
    if den == 1 && a == 2 * a0 {
      return Some((a0, period));
    }
  }
  None
}

/// Integer square root via binary search — exact on i128 inputs.
fn integer_sqrt(n: i128) -> i128 {
  if n < 0 {
    return 0;
  }
  if n < 2 {
    return n;
  }
  let mut lo: i128 = 0;
  let mut hi: i128 = (n as u128).isqrt() as i128;
  // Ensure hi*hi >= n
  while hi * hi < n {
    hi += 1;
  }
  while lo < hi {
    let mid = (lo + hi + 1) / 2;
    if mid * mid <= n {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  lo
}

/// ContinuedFraction[x] - exact continued fraction for rational numbers
/// ContinuedFraction[x, n] - first n terms for real numbers
/// Apply the term-count argument of the exact (Integer/Rational) forms: with a
/// positive integer `n`, truncate the terms when longer than `n` and, when the
/// finite continued fraction is shorter than `n`, emit wolframscript's
/// `ContinuedFraction::incomp` warning. The 1-argument form is unchanged.
fn finalize_exact_continued_fraction(
  mut terms: Vec<Expr>,
  args: &[Expr],
) -> Expr {
  if args.len() == 2
    && let Expr::Integer(n) = &args[1]
    && *n > 0
  {
    let n = *n as usize;
    if terms.len() > n {
      terms.truncate(n);
    } else if terms.len() < n {
      crate::emit_message(&format!(
        "ContinuedFraction::incomp: Warning: ContinuedFraction terminated before {n} terms."
      ));
    }
  }
  Expr::List(terms.into())
}

/// The first `n` continued-fraction terms of an exact quadratic irrational —
/// `Sqrt[d]`, `GoldenRatio`, or `(p + q Sqrt[d]) / r` — taken from its
/// eventually periodic expansion, or `None` when `expr` is not one.
fn exact_quadratic_cf_terms(expr: &Expr, n: usize) -> Option<Vec<i128>> {
  let (pre, period) = if let Some(d) = extract_sqrt_integer(expr) {
    if d <= 0 {
      return None;
    }
    let (a0, period) = continued_fraction_of_sqrt(d)?;
    (vec![a0], period)
  } else if matches!(expr, Expr::Identifier(s) | Expr::Constant(s) if s == "GoldenRatio")
  {
    continued_fraction_of_quadratic(1, 1, 5, 2)?
  } else {
    let (p, q, d, r) = extract_quadratic_irrational(expr)?;
    if d <= 0 || q == 0 || r == 0 || is_perfect_square(d) {
      return None;
    }
    continued_fraction_of_quadratic(p, q, d, r)?
  };

  let mut terms = Vec::with_capacity(n);
  for value in pre {
    if terms.len() == n {
      return Some(terms);
    }
    terms.push(value);
  }
  if period.is_empty() {
    // A rational value: the expansion simply ends.
    return Some(terms);
  }
  while terms.len() < n {
    for value in &period {
      if terms.len() == n {
        break;
      }
      terms.push(*value);
    }
  }
  Some(terms)
}

pub fn continued_fraction_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "ContinuedFraction expects 1 or 2 arguments".into(),
    ));
  }

  // Handle Rational[p, q] or Integer
  match &args[0] {
    Expr::Integer(n) => {
      return Ok(finalize_exact_continued_fraction(
        vec![Expr::Integer(*n)],
        args,
      ));
    }
    Expr::FunctionCall { name, args: rargs }
      if name == "Rational" && rargs.len() == 2 =>
    {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&rargs[0], &rargs[1]) {
        // Wolfram expands |x| and negates every quotient for negative
        // x (termwise negation), not the floor-based expansion:
        // ContinuedFraction[-1/2] is {0, -2}, not {-1, 2}
        let negative = (*p < 0) != (*q < 0);
        let mut result = Vec::new();
        let mut a = p.abs();
        let mut b = q.abs();
        while b != 0 {
          let quotient = a / b;
          result.push(Expr::Integer(if negative {
            -quotient
          } else {
            quotient
          }));
          let rem = a % b;
          a = b;
          b = rem;
        }
        return Ok(finalize_exact_continued_fraction(result, args));
      }
    }
    _ => {}
  }

  // Machine-precision Real argument. Wolfram returns only the terms justified
  // by the number's precision, which equal Drop[ContinuedFraction[Rationalize[x]], -1]
  // (e.g. ContinuedFraction[2.5] = {2}, ContinuedFraction[3.245] = {3, 4, 12}).
  // A whole-number Real (Rationalize gives an integer) is left unevaluated, as
  // is a non-positive/non-integer second argument.
  if matches!(&args[0], Expr::Real(_)) {
    let unevaluated = || Ok(unevaluated("ContinuedFraction", args));
    let n_cap: Option<usize> = if args.len() == 2 {
      match &args[1] {
        Expr::Integer(n) if *n > 0 => Some(*n as usize),
        _ => return unevaluated(),
      }
    } else {
      None
    };
    let mut rat = crate::evaluator::evaluate_function_call_ast(
      "Rationalize",
      std::slice::from_ref(&args[0]),
    )?;
    // A number with no nearby simple fraction stays a Real under the
    // default tolerance; then its exact binary value is expanded (and the
    // recursion below must never see a Real again).
    if matches!(rat, Expr::Real(_)) {
      rat = crate::evaluator::evaluate_function_call_ast(
        "Rationalize",
        &[args[0].clone(), Expr::Integer(0)],
      )?;
    }
    if matches!(rat, Expr::Integer(_) | Expr::Real(_)) {
      return unevaluated();
    }
    let full = continued_fraction_ast(&[rat])?;
    if let Expr::List(items) = &full {
      let mut terms: Vec<Expr> = items.iter().cloned().collect();
      // Drop the last (precision-unjustified) term.
      terms.pop();
      if terms.is_empty() {
        return unevaluated();
      }
      if let Some(n) = n_cap {
        if terms.len() > n {
          terms.truncate(n);
        } else if terms.len() < n {
          // The float's precision didn't justify n terms.
          crate::emit_message(&format!(
            "ContinuedFraction::incomp: Warning: ContinuedFraction terminated before {n} terms."
          ));
        }
      }
      return Ok(Expr::List(terms.into()));
    }
    return unevaluated();
  }

  // An arbitrary-precision number: the terms its precision determines, i.e.
  // the common prefix of the expansions of the ends of its uncertainty
  // interval x (1 ± 10^-precision).
  if let Expr::BigFloat(digits, precision) = &args[0] {
    let unevaluated = || Ok(unevaluated("ContinuedFraction", args));
    let n_cap: Option<usize> = match args.get(1) {
      None => None,
      Some(Expr::Integer(n)) if *n > 0 => Some(*n as usize),
      Some(_) => return unevaluated(),
    };
    let Some((num, den)) = decimal_to_big_rational(digits) else {
      return unevaluated();
    };
    let Ok(scale_digits) = u32::try_from(precision.ceil().max(1.0) as i64)
    else {
      return unevaluated();
    };
    use num_bigint::BigInt;
    let scale = BigInt::from(10).pow(scale_digits);
    // x (1 ± 1/scale) = num (scale ± 1) / (den scale)
    let lower = (&num * (&scale - 1), &den * &scale);
    let upper = (&num * (&scale + 1), &den * &scale);
    let (lo_terms, hi_terms) = (big_cf_terms(lower), big_cf_terms(upper));
    let mut terms: Vec<Expr> = lo_terms
      .iter()
      .zip(&hi_terms)
      .take_while(|(a, b)| a == b)
      .map(|(a, _)| crate::functions::math_ast::bigint_to_expr(a.clone()))
      .collect();
    if terms.is_empty() {
      return unevaluated();
    }
    if let Some(n) = n_cap {
      if terms.len() > n {
        terms.truncate(n);
      } else if terms.len() < n {
        crate::emit_message(&format!(
          "ContinuedFraction::incomp: Warning: ContinuedFraction terminated before {n} terms."
        ));
      }
    }
    return Ok(Expr::List(terms.into()));
  }

  // ContinuedFraction[Sqrt[D]] — periodic CF for non-square positive integers.
  // Note: Sqrt[n] is canonicalized internally to Power[n, Rational[1, 2]], so
  // this matcher reaches for the Power shape.
  let sqrt_arg: Option<i128> = if args.len() == 1 {
    extract_sqrt_integer(&args[0])
  } else {
    None
  };
  if let Some(d) = sqrt_arg
    && d > 0
    && let Some((a0, period)) = continued_fraction_of_sqrt(d)
  {
    if period.is_empty() {
      return Ok(Expr::List(vec![Expr::Integer(a0)].into()));
    }
    let period_list =
      Expr::List(period.into_iter().map(Expr::Integer).collect());
    return Ok(Expr::List(vec![Expr::Integer(a0), period_list].into()));
  }

  // GoldenRatio = (1 + √5) / 2
  if args.len() == 1
    && matches!(&args[0], Expr::Identifier(s) | Expr::Constant(s) if s == "GoldenRatio")
    && let Some((pre, period)) = continued_fraction_of_quadratic(1, 1, 5, 2)
  {
    let mut result: Vec<Expr> = pre.into_iter().map(Expr::Integer).collect();
    if !period.is_empty() {
      result.push(Expr::List(period.into_iter().map(Expr::Integer).collect()));
    }
    return Ok(Expr::List(result.into()));
  }

  // ContinuedFraction[(p + q √d) / r] — periodic CF for quadratic irrationals.
  if args.len() == 1
    && let Some((p, q, d, r)) = extract_quadratic_irrational(&args[0])
    && d > 0
    && q != 0
    && r != 0
    && !is_perfect_square(d)
    && let Some((pre, period)) = continued_fraction_of_quadratic(p, q, d, r)
  {
    let pre_exprs: Vec<Expr> = pre.into_iter().map(Expr::Integer).collect();
    let mut result: Vec<Expr> = pre_exprs;
    if !period.is_empty() {
      result.push(Expr::List(period.into_iter().map(Expr::Integer).collect()));
    }
    return Ok(Expr::List(result.into()));
  }

  // For expressions with n terms
  if args.len() == 2 {
    let n = match &args[1] {
      Expr::Integer(n) if *n > 0 => *n as usize,
      _ => {
        return Ok(unevaluated("ContinuedFraction", args));
      }
    };

    // A quadratic irrational has an eventually periodic expansion, so the
    // first n terms come out of the exact integer recurrence rather than a
    // float. `ContinuedFraction[Sqrt[2], 200]` is 200 exact 2s; the f64
    // fallback below drifts off the true expansion after ~20 terms because a
    // double only carries ~16 digits (Project Euler 57 walks 1000 of them).
    if let Some(terms) = exact_quadratic_cf_terms(&args[0], n) {
      return Ok(Expr::List(terms.into_iter().map(Expr::Integer).collect()));
    }

    // Try high-precision computation for known constants
    if let Some(cf) = try_constant_continued_fraction(&args[0], n) {
      return Ok(Expr::List(cf.into_iter().map(Expr::Integer).collect()));
    }

    // Fall back to f64 for generic real expressions
    if let Some(x) = try_eval_to_f64(&args[0]) {
      let mut result = Vec::new();
      let mut val = x;
      for _ in 0..n {
        let a = val.floor() as i128;
        result.push(Expr::Integer(a));
        let frac = val - a as f64;
        if frac.abs() < 1e-10 {
          break;
        }
        val = 1.0 / frac;
      }
      return Ok(Expr::List(result.into()));
    }
  }

  Ok(unevaluated("ContinuedFraction", args))
}

/// The exact value of a decimal digit string (`-12.345`, `1.5*^-20`) as a
/// numerator and positive denominator.
fn decimal_to_big_rational(
  digits: &str,
) -> Option<(num_bigint::BigInt, num_bigint::BigInt)> {
  use num_bigint::BigInt;
  let (mantissa, exponent) = match digits.split_once("*^") {
    Some((m, e)) => (m, e.parse::<i64>().ok()?),
    None => match digits.split_once(['e', 'E']) {
      Some((m, e)) => (m, e.parse::<i64>().ok()?),
      None => (digits, 0),
    },
  };
  let (int_part, frac_part) =
    mantissa.split_once('.').unwrap_or((mantissa, ""));
  let all: String = format!("{int_part}{frac_part}");
  let mut num: BigInt = all.parse().ok()?;
  let mut den = BigInt::from(1);
  let shift = exponent - frac_part.len() as i64;
  let ten = BigInt::from(10);
  if shift >= 0 {
    num *= ten.pow(u32::try_from(shift).ok()?);
  } else {
    den = ten.pow(u32::try_from(-shift).ok()?);
  }
  Some((num, den))
}

/// The continued-fraction terms of the rational `num/den` (den > 0), floor
/// based. Capped so a huge denominator cannot run away.
fn big_cf_terms(
  (mut num, mut den): (num_bigint::BigInt, num_bigint::BigInt),
) -> Vec<num_bigint::BigInt> {
  let mut out = Vec::new();
  while !den.is_zero() && out.len() < 100_000 {
    // Floor division (the denominator stays positive).
    let mut q = &num / &den;
    let mut r = &num - &q * &den;
    if r.is_negative() {
      q -= 1;
      r += &den;
    }
    out.push(q);
    num = den;
    den = r;
  }
  out
}

/// Exact rational with i128 numerator/denominator, kept reduced with a
/// positive denominator. Used to evaluate a periodic continued fraction in
/// the field Q(sqrt(disc)).
#[derive(Clone, Copy)]
struct Rat {
  num: i128,
  den: i128,
}

impl Rat {
  fn new(num: i128, den: i128) -> Self {
    let mut r = Self { num, den };
    r.reduce();
    r
  }
  fn int(n: i128) -> Self {
    Self { num: n, den: 1 }
  }
  fn reduce(&mut self) {
    (self.num, self.den) = rat_reduce(self.num, self.den);
  }
  fn add(self, o: Self) -> Self {
    Self::new(self.num * o.den + o.num * self.den, self.den * o.den)
  }
  fn mul(self, o: Self) -> Self {
    Self::new(self.num * o.num, self.den * o.den)
  }
  fn div(self, o: Self) -> Self {
    Self::new(self.num * o.den, self.den * o.num)
  }
  fn is_zero(self) -> bool {
    self.num == 0
  }
}

/// Factor the largest square out of `n >= 0`: returns (f, d) with n = f^2 * d
/// and d square-free.
fn pull_square_factor(n: i128) -> (i128, i128) {
  let mut d = n;
  let mut f = 1i128;
  let mut i = 2i128;
  while i * i <= d {
    while d % (i * i) == 0 {
      d /= i * i;
      f *= i;
    }
    i += 1;
  }
  (f, d)
}

/// Evaluate a periodic continued fraction `[prefix; {period}]` to its exact
/// quadratic-surd value, returned as a Woxi expression in WL's canonical
/// `(P + S Sqrt[D])/Q` form. Returns None if the (small-integer) arithmetic
/// degenerates or would overflow into a non-surd.
fn periodic_continued_fraction(
  prefix: &[i128],
  period: &[i128],
) -> Option<Expr> {
  // Convergents h_i/k_i of the finite period block, with the standard
  // h_{-2}=0, h_{-1}=1, k_{-2}=1, k_{-1}=0 seeds.
  let (mut h_prev2, mut h_prev1) = (0i128, 1i128);
  let (mut k_prev2, mut k_prev1) = (1i128, 0i128);
  for &p in period {
    let h = p.checked_mul(h_prev1)?.checked_add(h_prev2)?;
    h_prev2 = h_prev1;
    h_prev1 = h;
    let k = p.checked_mul(k_prev1)?.checked_add(k_prev2)?;
    k_prev2 = k_prev1;
    k_prev1 = k;
  }
  let (hm1, hm2, km1, km2) = (h_prev1, h_prev2, k_prev1, k_prev2);

  // The purely periodic value y satisfies A y^2 + B y + C = 0 with
  // A = k_{m-1}, B = k_{m-2} - h_{m-1}, C = -h_{m-2}.
  let a_co = km1;
  let b_co = km2 - hm1;
  let c_co = -hm2;
  let disc = b_co * b_co - 4 * a_co * c_co;
  if disc <= 0 || a_co == 0 {
    return None;
  }
  // y = (-B + sqrt(disc)) / (2A) = acc_a + acc_b * sqrt(disc).
  let two_a = 2 * a_co;
  let mut acc_a = Rat::new(-b_co, two_a);
  let mut acc_b = Rat::new(1, two_a);

  // Fold the non-periodic prefix in from the right: acc -> a_i + 1/acc.
  let disc_rat = Rat::int(disc);
  for &t in prefix.iter().rev() {
    // 1/(a + b sqrt(disc)) = (a - b sqrt(disc)) / (a^2 - b^2 disc).
    let norm = acc_a
      .mul(acc_a)
      .add(acc_b.mul(acc_b).mul(disc_rat).mul(Rat::int(-1)));
    if norm.is_zero() {
      return None;
    }
    let inv_a = acc_a.div(norm);
    let inv_b = acc_b.div(norm).mul(Rat::int(-1));
    acc_a = inv_a.add(Rat::int(t));
    acc_b = inv_b;
  }

  // value = acc_a + acc_b * sqrt(disc); render as (P + S Sqrt[D])/Q.
  let q = acc_a.den / gcd_i128(acc_a.den, acc_b.den) * acc_b.den; // lcm
  let pn = acc_a.num * (q / acc_a.den);
  let rn = acc_b.num * (q / acc_b.den);
  if rn == 0 {
    // Degenerate to a rational (shouldn't happen for a genuine periodic CF).
    let (pn, q) = rat_reduce(pn, q);
    return Some(if q == 1 {
      Expr::Integer(pn)
    } else {
      call("Rational", vec![Expr::Integer(pn), Expr::Integer(q)])
    });
  }
  // rn*sqrt(disc) = sign(rn) * sqrt(rn^2 * disc) = sign(rn) * f * sqrt(d).
  let inner = rn.checked_mul(rn)?.checked_mul(disc)?;
  let (f, d) = pull_square_factor(inner);
  let s = rn.signum() * f;
  if d == 1 {
    // Perfect square under the root: the value is rational.
    let total = pn + s;
    let (total, q) = rat_reduce(total, q);
    return Some(if q == 1 {
      Expr::Integer(total)
    } else {
      call("Rational", vec![Expr::Integer(total), Expr::Integer(q)])
    });
  }
  // Reduce (P, S, Q) by their common factor so D stays square-free and the
  // fraction is in lowest terms, matching wolframscript.
  let g = gcd_i128(gcd_i128(pn, s), q);
  let (pn, s, q) = (pn / g, s / g, q / g);

  // Build (pn + s*Sqrt[d])/q and let the evaluator canonicalize the surd.
  let sqrt_d = call1("Sqrt", Expr::Integer(d));
  let s_sqrt = if s == 1 {
    sqrt_d
  } else {
    times(vec![Expr::Integer(s), sqrt_d])
  };
  let numer = if pn == 0 {
    s_sqrt
  } else {
    plus(vec![Expr::Integer(pn), s_sqrt])
  };
  let value = if q == 1 {
    numer
  } else {
    times(vec![numer, pow(Expr::Integer(q), Expr::Integer(-1))])
  };
  crate::evaluator::evaluate_expr_to_expr(&value).ok()
}

/// FromContinuedFraction[{a0, a1, a2, ...}] - reconstruct a number from its continued fraction
pub fn from_continued_fraction_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "FromContinuedFraction expects 1 argument".into(),
    ));
  }

  let Expr::List(elements) = &args[0] else {
    return Ok(unevaluated("FromContinuedFraction", args));
  };

  if elements.is_empty() {
    return Ok(id_expr("Infinity"));
  }

  // Periodic continued fraction: the last element is a sublist holding the
  // repeating block, e.g. {1, {2}} -> Sqrt[2], {{1}} -> GoldenRatio. The
  // value is a quadratic surd. All other elements must be the integer
  // non-periodic prefix.
  if let Some(Expr::List(period)) = elements.last() {
    let prefix: Option<Vec<i128>> = elements[..elements.len() - 1]
      .iter()
      .map(|e| match e {
        Expr::Integer(n) => Some(*n),
        _ => None,
      })
      .collect();
    let per: Option<Vec<i128>> = period
      .iter()
      .map(|e| match e {
        Expr::Integer(n) => Some(*n),
        _ => None,
      })
      .collect();
    match (prefix, per) {
      (Some(prefix), Some(per)) if !per.is_empty() => {
        if let Some(expr) = periodic_continued_fraction(&prefix, &per) {
          return Ok(expr);
        }
      }
      _ => {}
    }
    return Ok(unevaluated("FromContinuedFraction", args));
  }

  // Collect all integers. Convergents of even moderately long continued
  // fractions (e.g. `ContinuedFraction[E, 100]`) overflow i128 long before
  // the last term, so accumulate in `BigInt`.
  let mut ints: Vec<BigInt> = Vec::new();
  for elem in elements {
    if let Some(n) = expr_to_bigint(elem) {
      ints.push(n);
    } else {
      // Symbolic terms: fold from the right, combining into a single
      // fraction at every step (the convergent recurrence).
      if elements.iter().any(|e| matches!(e, Expr::List(_))) {
        return Ok(unevaluated("FromContinuedFraction", args));
      }
      let mut acc = elements.last().unwrap().clone();
      for e in elements[..elements.len() - 1].iter().rev() {
        let step = Expr::FunctionCall {
          name: "Together".to_string(),
          args: vec![Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![
              e.clone(),
              Expr::FunctionCall {
                name: "Power".to_string(),
                args: vec![acc, Expr::Integer(-1)].into(),
              },
            ]
            .into(),
          }]
          .into(),
        };
        acc = crate::evaluator::evaluate_expr_to_expr(&step)?;
      }
      return Ok(acc);
    }
  }

  // Build fraction from right to left: start with last element, then a_i + 1/acc
  // Use numerator/denominator representation to stay exact
  let mut num = ints.last().unwrap().clone();
  let mut den = BigInt::from(1);

  for i in (0..ints.len() - 1).rev() {
    // acc = num/den, we want ints[i] + 1/acc = ints[i] + den/num = (ints[i]*num + den)/num
    let new_num = &ints[i] * &num + &den;
    den = num;
    num = new_num;
  }

  Ok(crate::functions::math_ast::number_theory::make_rational_expr(&num, &den))
}

/// IntegerDigits[n], IntegerDigits[n, b], IntegerDigits[n, b, len]
pub fn integer_digits_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "IntegerDigits expects 1 to 3 arguments".into(),
    ));
  }

  let unevaluated = || unevaluated("IntegerDigits", args);
  let show =
    |e: &Expr| crate::syntax::format_expr(e, crate::syntax::ExprForm::Output);

  // Position 1: an explicit non-integer number emits ::int; symbolic
  // subjects stay silently unevaluated.
  let is_integer_subject =
    matches!(&args[0], Expr::Integer(_) | Expr::BigInteger(_));
  if !is_integer_subject {
    let is_explicit_non_integer =
      matches!(&args[0], Expr::Real(_) | Expr::BigFloat(..))
        || matches!(&args[0], Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2);
    if is_explicit_non_integer {
      crate::emit_message(&format!(
        "IntegerDigits::int: Integer expected at position 1 in {}.",
        show(&unevaluated())
      ));
    }
    return Ok(unevaluated());
  }

  // MixedRadix base: `IntegerDigits[n, MixedRadix[{r0, ..., r_{m-1}}]]` gives the
  // digits of n in the mixed-radix number system. The least-significant digit is
  // `n mod r_{m-1}`, the next `n mod r_{m-2}`, …, and an implicit leading place
  // holds whatever remains. Leading zeros are stripped (a single 0 is kept for
  // n == 0). A negative subject uses its absolute value, like the integer-base
  // form. The inverse of `FromDigits[_, MixedRadix[_]]`.
  if args.len() >= 2
    && let Expr::FunctionCall {
      name: mr,
      args: mr_args,
    } = &args[1]
    && mr == "MixedRadix"
    && mr_args.len() == 1
    && let Expr::List(radices) = &mr_args[0]
  {
    let radix_vals = match radices
      .iter()
      .map(expr_to_bigint)
      .collect::<Option<Vec<_>>>()
    {
      Some(v) if v.iter().all(|r| *r >= BigInt::from(1)) => v,
      _ => return Ok(unevaluated()),
    };
    let mut num = match expr_to_bigint(&args[0]) {
      Some(n) => n.abs(),
      None => return Ok(unevaluated()),
    };
    // Divide by each radix from least to most significant, then take the
    // remaining value as the leading digit (an unbounded leading place).
    let mut digits_lsf: Vec<BigInt> = Vec::with_capacity(radix_vals.len() + 1);
    for r in radix_vals.iter().rev() {
      digits_lsf.push(&num % r);
      num /= r;
    }
    digits_lsf.push(num);
    let mut digits: Vec<Expr> =
      digits_lsf.into_iter().rev().map(bigint_to_expr).collect();
    // Strip leading zeros, keeping at least one digit.
    while digits.len() > 1 && matches!(&digits[0], Expr::Integer(0)) {
      digits.remove(0);
    }
    // Optional length: pad with zeros on the left or keep the last `len` digits.
    if args.len() == 3
      && let Some(len) = expr_to_i128(&args[2])
      && len >= 0
    {
      let len = len as usize;
      if digits.len() < len {
        let mut padded = vec![Expr::Integer(0); len - digits.len()];
        padded.append(&mut digits);
        digits = padded;
      } else if digits.len() > len {
        digits = digits[digits.len() - len..].to_vec();
      }
    }
    return Ok(Expr::List(digits.into()));
  }

  // Position 2: an explicit base below 2 (or a non-integer number)
  // emits ::ibase; symbolic bases stay silently unevaluated.
  if args.len() >= 2 {
    let base_ok = matches!(&args[1], Expr::Integer(b) if *b >= 2);
    if !base_ok {
      let is_explicit_bad_base = matches!(
        &args[1],
        Expr::Integer(_)
          | Expr::BigInteger(_)
          | Expr::Real(_)
          | Expr::BigFloat(..)
      ) || matches!(&args[1], Expr::FunctionCall { name, args }
          if name == "Rational" && args.len() == 2);
      if is_explicit_bad_base {
        crate::emit_message(&format!(
          "IntegerDigits::ibase: Base {} is not an integer greater than 1.",
          show(&args[1])
        ));
      }
      return Ok(unevaluated());
    }
  }

  // Position 3: a non-negative machine integer; explicit other numbers
  // emit ::intnm; symbolic lengths stay silently unevaluated.
  if args.len() == 3 {
    let len_ok = matches!(&args[2], Expr::Integer(n) if (0..=i64::MAX as i128).contains(n));
    if !len_ok {
      let is_explicit_bad_len = matches!(
        &args[2],
        Expr::Integer(_)
          | Expr::BigInteger(_)
          | Expr::Real(_)
          | Expr::BigFloat(..)
      ) || matches!(&args[2], Expr::FunctionCall { name, args }
          if name == "Rational" && args.len() == 2);
      if is_explicit_bad_len {
        crate::emit_message(&format!(
          "IntegerDigits::intnm: Non-negative machine-sized integer expected at position 3 in {}.",
          show(&unevaluated())
        ));
      }
      return Ok(unevaluated());
    }
  }

  // Fast path: small integer input + small/default integer base — avoid
  // BigInt allocations for typical 1..i128 sized values.
  if let Expr::Integer(n_i128) = &args[0] {
    let base_i128: i128 = if args.len() >= 2 {
      match expr_to_i128(&args[1]) {
        Some(b) if b >= 2 => b,
        _ => 0, // unreachable after validation; signal fallback
      }
    } else {
      10
    };
    if base_i128 >= 2 {
      let mut num = n_i128.unsigned_abs();
      let base_u = base_i128 as u128;
      let mut digits: Vec<Expr> = Vec::with_capacity(20);
      if num == 0 {
        digits.push(Expr::Integer(0));
      } else {
        while num != 0 {
          digits.push(Expr::Integer((num % base_u) as i128));
          num /= base_u;
        }
        digits.reverse();
      }
      if args.len() == 3
        && let Some(len) = expr_to_i128(&args[2])
        && len >= 0
      {
        let len = len as usize;
        if digits.len() < len {
          let mut padded = vec![Expr::Integer(0); len - digits.len()];
          padded.append(&mut digits);
          digits = padded;
        } else if digits.len() > len {
          digits = digits[digits.len() - len..].to_vec();
        }
      }
      return Ok(Expr::List(digits.into()));
    }
  }

  let n = match expr_to_bigint(&args[0]) {
    Some(n) => n.abs(),
    None => {
      return Ok(unevaluated());
    }
  };

  let base = if args.len() >= 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => BigInt::from(b),
      _ => return Ok(unevaluated()),
    }
  } else {
    BigInt::from(10)
  };
  let mut digits = Vec::new();
  if n.is_zero() {
    digits.push(Expr::Integer(0));
  } else {
    let mut num = n;
    while !num.is_zero() {
      digits.push(bigint_to_expr(&num % &base));
      num /= &base;
    }
    digits.reverse();
  }

  // Handle optional length parameter
  if args.len() == 3
    && let Some(len) = expr_to_i128(&args[2])
    && len >= 0
  {
    let len = len as usize;
    if digits.len() < len {
      // Pad with zeros on the left
      let mut padded = vec![Expr::Integer(0); len - digits.len()];
      padded.append(&mut digits);
      digits = padded;
    } else if digits.len() > len {
      // Truncate from the left (keep least significant digits)
      digits = digits[digits.len() - len..].to_vec();
    }
  }

  Ok(Expr::List(digits.into()))
}

/// Extract digits of a positive BigFloat in an arbitrary integer `base >= 2`,
/// returning `(digits, exponent)` where:
/// - `digits[0]` is the most-significant digit.
/// - `exponent` is the number of digits left of the radix point (Wolfram's
///   `RealDigits` convention), so `Pi` in base 260 gives exponent `1` and
///   `0.05` in base 10 gives exponent `-1`.
///
/// Produces exactly `num_digits` digits.
fn bigfloat_to_digits_base(
  bf: &astro_float::BigFloat,
  base: i128,
  num_digits: usize,
) -> Result<(Vec<i128>, i64), InterpreterError> {
  use num_bigint::BigUint;

  if bf.is_zero() {
    return Ok((vec![0; num_digits.max(1)], 0));
  }
  if base < 2 {
    return Err(InterpreterError::EvaluationError(
      "RealDigits: base must be >= 2".into(),
    ));
  }
  let raw_err = || {
    InterpreterError::EvaluationError(
      "RealDigits: cannot read the number's bits".into(),
    )
  };
  // The BigFloat is exactly `0.m × 2^exp` with `m` the little-endian words;
  // rewrite it as the exact fraction `num / 2^k` and do everything in integer
  // arithmetic, so the cost is one big division per digit group instead of a
  // full-precision float operation per digit.
  let (words, _, _, exp, _) = bf.as_raw_parts().ok_or_else(raw_err)?;
  let mut mantissa = BigUint::zero();
  for &w in words.iter().rev() {
    mantissa = (mantissa << usize::BITS) | BigUint::from(w);
  }
  let pow2 = exp as i64 - (words.len() * usize::BITS as usize) as i64;
  let (num, den_shift) = if pow2 >= 0 {
    (mantissa << pow2 as usize, 0usize)
  } else {
    (mantissa, (-pow2) as usize)
  };
  let big_base = BigUint::from(base as u128);

  // Find `shift` with base^shift <= x < base^(shift+1): estimate it from the
  // binary magnitude, then correct with exact comparisons.
  let log2_base = (base as f64).log2();
  let mut shift = (((num.bits() as i64 - den_shift as i64) as f64 - 1.0)
    / log2_base)
    .floor() as i64;
  // `x * base^-shift` as an exact comparison against 1 and `base`.
  let ge_pow = |s: i64| -> bool {
    // x >= base^s  <=>  num >= base^s * 2^den_shift   (s >= 0)
    //              <=>  num * base^-s >= 2^den_shift  (s < 0)
    if s >= 0 {
      num >= (big_base.pow(s as u32) << den_shift)
    } else {
      (&num * big_base.pow((-s) as u32)) >= (BigUint::from(1u8) << den_shift)
    }
  };
  while ge_pow(shift + 1) {
    shift += 1;
  }
  while !ge_pow(shift) {
    shift -= 1;
  }
  // Wolfram's exponent = shift + 1: the most-significant digit sits at
  // position `shift`, so `shift + 1` digits lie left of the radix point.
  let exponent = shift + 1;

  // floor(x * base^(num_digits - 1 - shift)) has exactly `num_digits` digits.
  let scale = num_digits as i64 - 1 - shift;
  let mut scaled = num;
  let mut den = BigUint::from(1u8) << den_shift;
  if scale >= 0 {
    scaled *= big_base.pow(scale as u32);
  } else {
    den *= big_base.pow((-scale) as u32);
  }
  let mut value = scaled / den;

  let mut digits: Vec<i128> = Vec::with_capacity(num_digits);
  let base_u32 = u32::try_from(base).ok();
  match base_u32 {
    Some(b) => {
      // Peel off a group of digits at a time to keep the divisions few.
      let mut group = b as u64;
      let mut group_len = 1usize;
      while let Some(next) = group.checked_mul(b as u64)
        && u32::try_from(next).is_ok()
      {
        group = next;
        group_len += 1;
      }
      let group_big = BigUint::from(group);
      while !value.is_zero() {
        let (q, r) = (&value / &group_big, &value % &group_big);
        let mut r = r.iter_u64_digits().next().unwrap_or(0);
        for _ in 0..group_len {
          digits.push((r % b as u64) as i128);
          r /= b as u64;
        }
        value = q;
      }
    }
    None => {
      while !value.is_zero() {
        let (q, r) = (&value / &big_base, &value % &big_base);
        digits.push(r.to_u128().unwrap_or(0) as i128);
        value = q;
      }
    }
  }
  // Groups were emitted least-significant first and padded with zeros.
  digits.reverse();
  let lead = digits.iter().position(|&d| d != 0).unwrap_or(digits.len());
  digits.drain(..lead);
  digits.resize(num_digits, 0);
  Ok((digits, exponent))
}

/// Extract decimal digits and exponent from a BigFloat.
/// Returns (digit_chars, decimal_exponent) where digit_chars are ASCII digit bytes
/// and decimal_exponent is the number of integer digits (position of decimal point).
fn bigfloat_to_digits(
  bf: &astro_float::BigFloat,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<(Vec<u8>, i64), InterpreterError> {
  if bf.is_zero() {
    return Ok((vec![b'0'], 0));
  }

  let (_sign, digits, exponent) = bf
    .convert_to_radix(astro_float::Radix::Dec, rm, cc)
    .map_err(|e| {
      InterpreterError::EvaluationError(format!(
        "RealDigits: conversion error: {e}"
      ))
    })?;

  if digits.is_empty() || digits.iter().all(|&d| d == 0) {
    return Ok((vec![b'0'], 0));
  }

  // Convert digit values (0-9) to ASCII bytes
  let ascii_digits: Vec<u8> = digits.iter().map(|&d| b'0' + d).collect();

  Ok((ascii_digits, exponent as i64))
}

/// Convert a decimal-literal Real to an exact rational `(numerator, denominator)`
/// by parsing its {:?} representation. This preserves the decimal value the
/// user typed (e.g. 123.45 → (12345, 100)) rather than the binary-floating
/// approximation, which is what wolframscript's base-conversion digit
/// functions expect.
fn real_to_rational(r: f64) -> Option<(i128, i128)> {
  if !r.is_finite() {
    return None;
  }
  let s = format!("{r:?}");
  let sign = if r < 0.0 { -1i128 } else { 1 };
  let unsigned = s.trim_start_matches('-');

  if let Some(e_idx) = unsigned.find(['e', 'E']) {
    let (mant, exp_part) = unsigned.split_at(e_idx);
    let exp: i32 = exp_part[1..].parse().ok()?;
    let (n, d) = decimal_to_rational(mant)?;
    return apply_exponent(n, d, exp).map(|(n, d)| (sign * n, d));
  }
  let (n, d) = decimal_to_rational(unsigned)?;
  Some((sign * n, d))
}

fn decimal_to_rational(s: &str) -> Option<(i128, i128)> {
  if let Some(dot_idx) = s.find('.') {
    let int_part = &s[..dot_idx];
    let frac_part = &s[dot_idx + 1..];
    let combined = format!("{int_part}{frac_part}");
    let numer: i128 = combined.parse().ok()?;
    let denom: i128 = 10i128.checked_pow(frac_part.len() as u32)?;
    Some((numer, denom))
  } else {
    let numer: i128 = s.parse().ok()?;
    Some((numer, 1))
  }
}

fn apply_exponent(n: i128, d: i128, exp: i32) -> Option<(i128, i128)> {
  if exp >= 0 {
    let mul = 10i128.checked_pow(exp as u32)?;
    Some((n.checked_mul(mul)?, d))
  } else {
    let mul = 10i128.checked_pow((-exp) as u32)?;
    Some((n, d.checked_mul(mul)?))
  }
}

/// Compute RealDigits for a rational number in an arbitrary base.
fn real_digits_rational_base(
  numer: i128,
  denom: i128,
  base: i128,
) -> (Vec<Expr>, i128) {
  use std::collections::HashMap;

  let n = numer.abs();
  let d = denom.abs();

  // Compute exponent: number of digits in integer part
  let integer_part = n / d;

  let exponent: i128 = if integer_part == 0 {
    // For numbers < 1, exponent is 0 or negative
    // Count leading zeros after decimal point
    let mut temp = n * base;
    let mut leading_zeros: i128 = 0;
    while temp < d && temp > 0 {
      leading_zeros += 1;
      temp *= base;
    }
    -leading_zeros
  } else {
    // Count digits in integer part
    let mut count = 0i128;
    let mut temp = integer_part;
    while temp > 0 {
      count += 1;
      temp /= base;
    }
    count
  };

  // Normalize: bring remainder into [0, d) range and extract integer part digits
  let mut remainder = n;
  let int_part = remainder / d;
  remainder %= d;

  // Extract integer part digits
  let mut int_digits: Vec<i128> = Vec::new();
  if int_part > 0 {
    let mut temp = int_part;
    while temp > 0 {
      int_digits.push(temp % base);
      temp /= base;
    }
    int_digits.reverse();
  }

  // A pure fraction starts at its first nonzero digit: the leading zeros
  // are the (negative) exponent, not digits, and the recurring block is
  // counted from there (RealDigits[1/7, 2] is {{{1, 0, 0}}, -2}).
  if int_part == 0 {
    for _ in 0..(-exponent) {
      remainder *= base;
    }
  }

  // Now do long division for the fractional part
  // Track remainder -> position for cycle detection
  let mut digits: Vec<i128> = int_digits;
  let mut remainder_positions: HashMap<i128, usize> = HashMap::new();
  let mut cycle_start: Option<usize> = None;

  loop {
    if remainder == 0 {
      break;
    }

    if let Some(&pos) = remainder_positions.get(&remainder) {
      cycle_start = Some(pos);
      break;
    }

    remainder_positions.insert(remainder, digits.len());
    remainder *= base;
    let digit = remainder / d;
    remainder %= d;
    digits.push(digit);
  }

  // Build the result
  if let Some(cycle_pos) = cycle_start {
    let non_repeating: Vec<Expr> = digits[..cycle_pos]
      .iter()
      .map(|&d| Expr::Integer(d))
      .collect();
    let repeating: Vec<Expr> = digits[cycle_pos..]
      .iter()
      .map(|&d| Expr::Integer(d))
      .collect();

    let mut result = non_repeating;
    result.push(Expr::List(repeating.into()));
    (result, exponent)
  } else {
    let digit_exprs: Vec<Expr> =
      digits.iter().map(|&d| Expr::Integer(d)).collect();
    (digit_exprs, exponent)
  }
}

/// RealDigits[x, base, num_digits] — extract decimal digits of a real number.
/// Returns {digit_list, exponent}.
/// `RealDigits[x, b, n]` for an exact positive `x` and a non-integer real
/// base `b` (`GoldenRatio`, `Pi`): the greedy expansion — each digit the
/// largest multiple of the place value that fits — worked out at a precision
/// well beyond the requested digits, so an exact finite expansion
/// (`3 = φ^1 + φ^-2`) ends in exact zeros. None for anything else.
fn real_digits_irrational_base(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  let n = expr_to_i128(&args[2]).filter(|n| (1..=1000).contains(n))? as usize;
  let x = &args[0];
  if !crate::functions::predicate_ast::is_numeric_q(x)
    || crate::functions::math_ast::contains_inexact_real(x)
    || !try_eval_to_f64(x).is_some_and(|v| v > 0.0)
  {
    return None;
  }
  let prec = Expr::Integer(n as i128 + 60);
  let num = |e: Expr| {
    crate::evaluator::evaluate_function_call_ast("N", &[e, prec.clone()]).ok()
  };
  let eval = |name: &str, a: Vec<Expr>| {
    crate::evaluator::evaluate_function_call_ast(name, &a).ok()
  };
  let b = num(args[1].clone())?;
  let mut rest = num(x.clone())?;
  // Exponent: the number of base-b places before the point.
  let log = eval("Log", vec![b.clone(), rest.clone()])?;
  let mut exponent = expr_to_i128(&eval("Floor", vec![log])?)? + 1;
  // The floating logarithm can be one off at an exact power.
  let place = |k: i128| eval("Power", vec![b.clone(), Expr::Integer(k)]);
  let ge = |a: &Expr, c: &Expr| {
    matches!(
      eval("GreaterEqual", vec![a.clone(), c.clone()]),
      Some(Expr::Identifier(ref t)) if t == "True"
    )
  };
  if ge(&rest, &place(exponent)?) {
    exponent += 1;
  } else if !ge(&rest, &place(exponent - 1)?) {
    exponent -= 1;
  }
  let tiny = num(Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![Expr::Integer(10), Expr::Integer(-(n as i128) - 40)].into(),
  })?;
  let mut digits = Vec::with_capacity(n);
  for k in 1..=n {
    let pv = place(exponent - k as i128)?;
    if !ge(&rest, &eval("Times", vec![tiny.clone(), pv.clone()])?) {
      digits.push(Expr::Integer(0));
      continue;
    }
    let ratio = eval("Divide", vec![rest.clone(), pv.clone()])?;
    // A remainder within rounding of a whole place counts as that place.
    let d = expr_to_i128(&eval(
      "Floor",
      vec![eval("Plus", vec![ratio, tiny.clone()])?],
    )?)?;
    rest = eval(
      "Subtract",
      vec![rest, eval("Times", vec![Expr::Integer(d), pv])?],
    )?;
    if !ge(&rest, &Expr::Integer(0)) {
      rest = Expr::Integer(0);
    }
    digits.push(Expr::Integer(d));
  }
  Some(Ok(Expr::List(
    vec![Expr::List(digits.into()), Expr::Integer(exponent)].into(),
  )))
}

pub fn real_digits_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 4 {
    return Err(InterpreterError::EvaluationError(
      "RealDigits expects 1 to 4 arguments".into(),
    ));
  }
  let unevaluated_call = || Ok(unevaluated("RealDigits", args));
  // Optional 4th argument: the position of the most-significant returned digit
  // (as the exponent of `base`, so Pi at position −3 means the 10^−3 place).
  let start_pos: Option<i128> = if args.len() == 4 {
    match expr_to_i128(&args[3]) {
      Some(n) => Some(n),
      None => return unevaluated_call(),
    }
  } else {
    None
  };

  let base: i128 = if args.len() >= 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => b,
      _ => {
        // A number that is not a real above 1 is reported; a symbolic base
        // (and, for now, a non-integer one) leaves the call as it is.
        let numeric = crate::functions::predicate_ast::is_numeric_q(&args[1]);
        let above_one = try_eval_to_f64(&args[1]).is_some_and(|b| b > 1.0);
        if numeric && !above_one {
          crate::emit_message(&format!(
            "RealDigits::rbase: Base {} is not a real number greater than 1.",
            crate::syntax::format_expr(
              &args[1],
              crate::syntax::ExprForm::Output
            )
          ));
        }
        if numeric
          && above_one
          && args.len() == 3
          && let Some(digits) = real_digits_irrational_base(args)
        {
          return digits;
        }
        return unevaluated_call();
      }
    }
  } else {
    10
  };

  let explicit_num_digits = args.len() >= 3;

  let num_digits: usize = if explicit_num_digits {
    match expr_to_i128(&args[2]) {
      // No digits at all still tells the exponent.
      Some(0) => {
        let mut one = args.to_vec();
        one[2] = Expr::Integer(1);
        return Ok(match &real_digits_ast(&one)? {
          Expr::List(parts) if parts.len() == 2 => {
            Expr::List(vec![Expr::List(vec![].into()), parts[1].clone()].into())
          }
          other => other.clone(),
        });
      }
      Some(n) if n > 0 => n as usize,
      _ => {
        crate::emit_message(&format!(
          "RealDigits::intnm: Non-negative machine-sized integer expected at \
           position 3 in {}.",
          crate::syntax::format_expr(
            &unevaluated("RealDigits", args),
            crate::syntax::ExprForm::Output
          )
        ));
        return unevaluated_call();
      }
    }
  } else if matches!(&args[0], Expr::Real(_)) && base != 10 {
    // A machine-precision Real carries ~53 bits of information; express that
    // as a digit count in the target base: ceil(53 / log2(base)). For base 2
    // that's 53 (matching wolframscript); for base 10 the decimal path's 16
    // is used instead.
    let ratio = 53.0_f64 / (base as f64).log2();
    (ratio.ceil() as usize).max(1)
  } else if let Expr::BigFloat(_, prec) = &args[0] {
    // An arbitrary-precision number returns as many digits as it carries,
    // in the target base: RealDigits[N[Pi, 20]] lists 20, and a 25-digit
    // number has 83 binary or 21 hexadecimal digits.
    ((prec * 10f64.ln() / (base as f64).ln()).round() as usize).max(1)
  } else {
    // Default: use machine-precision (~16 digits)
    16
  };

  let expr = &args[0];

  if crate::functions::predicate_ast::is_complex_number(expr) {
    crate::emit_message(&crate::syntax::format_message_with_expr(
      "RealDigits::realx: The value ",
      expr,
      " is not a real number.",
    ));
    return Ok(unevaluated("RealDigits", args));
  }

  // An exact irrational (Pi, Sqrt[2], Log[2]) has no natural digit count:
  // without an explicit length wolframscript reports ::ndig and leaves the
  // call unevaluated.
  if !explicit_num_digits
    && !matches!(
      expr,
      Expr::Integer(_)
        | Expr::BigInteger(_)
        | Expr::Real(_)
        | Expr::BigFloat(..)
    )
    && expr_to_rational(expr).is_none()
    && crate::functions::predicate_ast::is_numeric_q(expr)
  {
    crate::emit_message(
      "RealDigits::ndig: The number of digits to return cannot be determined.",
    );
    return Ok(unevaluated("RealDigits", args));
  }

  // An inexact zero has no digits to give: its exponent is where the
  // smallest machine number would sit, trunc(-1075 Log[base, 2]) (-323 in
  // base 10, -1075 in base 2), and an explicit count of its digits is that
  // many Indeterminates, as in wolframscript.
  let inexact_zero = match expr {
    Expr::Real(f) => *f == 0.0,
    Expr::BigFloat(..) => try_eval_to_f64(expr) == Some(0.0),
    _ => false,
  };
  // From a given position a zero is all zero digits, the first of them at
  // that position.
  if let Some(start) = start_pos
    && (inexact_zero || matches!(expr, Expr::Integer(0)))
  {
    return Ok(Expr::List(
      vec![
        Expr::List(vec![Expr::Integer(0); num_digits].into()),
        Expr::Integer(start + 1),
      ]
      .into(),
    ));
  }
  if inexact_zero && start_pos.is_none() {
    let exponent = (-1075.0 * 2f64.ln() / (base as f64).ln()).trunc() as i128;
    let digits: Vec<Expr> = if explicit_num_digits {
      vec![Expr::Identifier("Indeterminate".into()); num_digits]
    } else {
      vec![Expr::Integer(0)]
    };
    return Ok(Expr::List(
      vec![Expr::List(digits.into()), Expr::Integer(exponent)].into(),
    ));
  }

  // Determine sign and work with absolute value
  let is_negative = match expr {
    Expr::Integer(n) => *n < 0,
    Expr::Real(f) => *f < 0.0,
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      ..
    } => true,
    _ => false,
  };

  let abs_expr = if is_negative {
    call1("Abs", expr.clone())
  } else {
    expr.clone()
  };

  // Check for exact zero
  let is_zero = matches!(&abs_expr, Expr::Integer(0))
    || matches!(&abs_expr, Expr::Real(f) if *f == 0.0);

  if is_zero {
    let count = if explicit_num_digits { num_digits } else { 1 };
    let exp = i128::from(!explicit_num_digits);
    let digits = vec![Expr::Integer(0); count];
    return Ok(Expr::List(
      vec![Expr::List(digits.into()), Expr::Integer(exp)].into(),
    ));
  }

  // For Real inputs, promote the Real to an exact rational via its decimal
  // literal and reuse the rational digit path. This preserves the digits the
  // user typed (e.g. 123.55555 → 1,2,3,5,5,5,5,5,0,…) rather than exposing
  // the f64 rounding tail (…5,4,9,9,9,…) that the astro-float path produces
  // from the raw bit pattern.
  let rational_from_real = if let Expr::Real(r) = &abs_expr {
    real_to_rational(r.abs())
  } else {
    None
  };

  // For Reals promoted to rational, force the explicit-num_digits code path
  // below so the digit list is padded to machine precision (matching
  // wolframscript's fixed-width output for Real inputs).
  let treat_as_explicit = rational_from_real.is_some() || explicit_num_digits;

  // For rationals, use long division with cycle detection
  if let Some((numer, denom)) =
    rational_from_real.or_else(|| expr_to_rational(&abs_expr))
    && denom != 0
  {
    if treat_as_explicit {
      // With explicit num_digits: use long division and produce exactly
      // num_digits significant digits starting from the first nonzero.
      // The rational long division already gives us digits starting
      // from the most significant, but the integer part may be 0 so
      // we need to generate enough fractional digits.
      let mut remainder = numer.abs();
      let d = denom.abs();
      let int_part = remainder / d;
      remainder %= d;

      // Determine exponent from integer part (Wolfram convention: number
      // of digits to the left of the decimal point, so first digit sits
      // at position `exponent - 1`).
      let exponent: i128 = if int_part == 0 {
        let mut temp = remainder * base;
        let mut leading_zeros: i128 = 0;
        while temp < d && temp > 0 {
          leading_zeros += 1;
          temp *= base;
        }
        -leading_zeros
      } else {
        let mut count = 0i128;
        let mut temp = int_part;
        while temp > 0 {
          count += 1;
          temp /= base;
        }
        count
      };

      // When the caller provides `start_pos = p`, we need to collect
      // digits at positions p, p-1, …, p-num_digits+1. Pre-extract more
      // digits than `num_digits` so the skip-and-pad logic below can
      // align the slice correctly. The MSD sits at position `exponent - 1`.
      let (effective_digits, leading_pad) = if let Some(p) = start_pos {
        let msd_pos = exponent - 1;
        let skip = msd_pos - p; // > 0 when p < MSD (need extra digits)
        if skip >= 0 {
          ((skip as usize) + num_digits, 0usize)
        } else {
          // p above MSD: pad zeros on the left.
          (
            num_digits.saturating_sub((-skip) as usize),
            (-skip) as usize,
          )
        }
      } else {
        (num_digits, 0usize)
      };

      // Extract integer part digits
      let mut digits: Vec<i128> = Vec::new();
      if int_part > 0 {
        let mut temp = int_part;
        let mut int_digs = Vec::new();
        while temp > 0 {
          int_digs.push(temp % base);
          temp /= base;
        }
        int_digs.reverse();
        digits.extend(int_digs);
      } else {
        // Skip leading fractional zeros (they're accounted for in exponent)
        while remainder > 0 && remainder * base < d {
          remainder *= base;
        }
      }

      // Machine-precision cap: a Real input carries ~53 bits of information,
      // i.e. ceil(53 / log2(base)) digits in the target base (≈16 for base
      // 10). Any digit beyond that whose value isn't pinned down by an exact
      // (terminating) expansion is unknowable, and wolframscript marks it
      // `Indeterminate`.
      let precision_cap = if rational_from_real.is_some() {
        Some(((53.0_f64 / (base as f64).log2()).ceil() as usize).max(1))
      } else {
        None
      };
      let max_known = precision_cap
        .unwrap_or(effective_digits)
        .min(effective_digits);

      // Generate up to `max_known` known digits.
      while digits.len() < max_known {
        remainder *= base;
        let digit = remainder / d;
        remainder %= d;
        digits.push(digit);
      }
      digits.truncate(max_known);

      // Slice + pad to satisfy `start_pos` semantics.
      if let Some(p) = start_pos {
        let msd_pos = exponent - 1;
        let skip = msd_pos - p;
        if skip > 0 {
          let sk = skip as usize;
          if sk >= digits.len() {
            digits.clear();
          } else {
            digits.drain(..sk);
          }
        }
        // Trailing zeros to reach num_digits.
        while digits.len() + leading_pad < num_digits {
          digits.push(0);
        }
        if digits.len() + leading_pad > num_digits {
          digits.truncate(num_digits - leading_pad);
        }
        let mut digit_exprs: Vec<Expr> =
          std::iter::repeat_n(0i128, leading_pad)
            .chain(digits.iter().copied())
            .map(Expr::Integer)
            .collect();
        digit_exprs.truncate(num_digits);
        return Ok(Expr::List(
          vec![Expr::List(digit_exprs.into()), Expr::Integer(p + 1)].into(),
        ));
      }

      let mut digit_exprs: Vec<Expr> =
        digits.iter().map(|&dig| Expr::Integer(dig)).collect();
      // Pad beyond machine precision with Indeterminate. (Digits within the
      // precision cap that are 0 because the value terminated were already
      // emitted as 0 by the loop above; this only fires when the caller
      // explicitly requests more digits than the precision supports.)
      while digit_exprs.len() < num_digits {
        digit_exprs.push(id_expr("Indeterminate"));
      }
      return Ok(Expr::List(
        vec![Expr::List(digit_exprs.into()), Expr::Integer(exponent)].into(),
      ));
    }
    let (digit_list, exponent) = real_digits_rational_base(numer, denom, base);
    return Ok(Expr::List(
      vec![Expr::List(digit_list.into()), Expr::Integer(exponent)].into(),
    ));
  }

  // Compute with extra precision to avoid rounding errors in the last digits.
  // When a start position is given we also need enough precision to reach it.
  let leading_skip = start_pos.map(|p| {
    // We need digits from the MSD down to position p - num_digits + 1. A pure
    // upper bound here is fine; it will be resliced below.
    (num_digits as i128 + 16).max(-(p) + num_digits as i128 + 16) as usize
  });
  let precision = num_digits + leading_skip.unwrap_or(0) + 10;

  use astro_float::{Consts, RoundingMode};
  let mut cc = Consts::new().map_err(|e| {
    InterpreterError::EvaluationError(format!("BigFloat init error: {e}"))
  })?;
  let rm = RoundingMode::ToEven;
  let bits = nominal_bits(precision);

  let Ok(bf) = expr_to_bigfloat(&abs_expr, bits, rm, &mut cc) else {
    // Non-numeric expression (e.g. bare symbol): return unevaluated.
    return Ok(unevaluated("RealDigits", args));
  };

  // Non-decimal bases route through the arbitrary-base digit extractor.
  if base != 10 {
    // We need enough raw digits to cover both leading zeros skipped by
    // `start_pos` (when < MSD) and the requested count. The normalization
    // loop also benefits from a slightly larger precision window.
    let extract_count = num_digits + leading_skip.unwrap_or(0) + 4;
    let (raw, base_exp) = bigfloat_to_digits_base(&bf, base, extract_count)?;
    let mut digits = raw;
    if let Some(p) = start_pos {
      // digit k (0-indexed) sits at position `base_exp - 1 - k`. Skip
      // forward to start at position `p`, or pad with zeros if `p` is
      // above the MSD.
      let skip = base_exp as i128 - 1 - p;
      if skip > 0 {
        let sk = skip as usize;
        if sk >= digits.len() {
          digits.clear();
        } else {
          digits.drain(..sk);
        }
      } else if skip < 0 {
        let pad = (-skip) as usize;
        let mut padded = vec![0i128; pad];
        padded.extend(digits);
        digits = padded;
      }
      while digits.len() < num_digits {
        digits.push(0);
      }
      digits.truncate(num_digits);
      let digit_exprs: Vec<Expr> =
        digits.iter().map(|&d| Expr::Integer(d)).collect();
      return Ok(Expr::List(
        vec![Expr::List(digit_exprs.into()), Expr::Integer(p + 1)].into(),
      ));
    }
    let base_exp = base_exp as i128;
    while digits.len() < num_digits {
      digits.push(0);
    }
    digits.truncate(num_digits);
    let digit_exprs: Vec<Expr> =
      digits.iter().map(|&d| Expr::Integer(d)).collect();
    return Ok(Expr::List(
      vec![Expr::List(digit_exprs.into()), Expr::Integer(base_exp)].into(),
    ));
  }

  let (raw_digits, decimal_exp) = bigfloat_to_digits(&bf, rm, &mut cc)?;

  // raw_digits are the significant digits, decimal_exp is the exponent
  // (number of digits before the decimal point).
  let mut digits: Vec<i128> = raw_digits
    .iter()
    .filter(|b| b.is_ascii_digit())
    .map(|b| (*b - b'0') as i128)
    .collect();

  if let Some(p) = start_pos {
    // Skip leading digits so that the first returned digit is at position `p`.
    // digit k (0-indexed) is at position (decimal_exp as i128) - 1 - k,
    // so to start at position p we skip (decimal_exp - 1 - p) digits.
    let skip = (decimal_exp as i128) - 1 - p;
    if skip > 0 {
      let sk = skip as usize;
      if sk >= digits.len() {
        digits.clear();
      } else {
        digits.drain(..sk);
      }
    } else if skip < 0 {
      // Requested start is above the most significant digit: pad with zeros.
      let pad = (-skip) as usize;
      let mut padded = vec![0i128; pad];
      padded.extend(digits);
      digits = padded;
    }
    while digits.len() < num_digits {
      digits.push(0);
    }
    digits.truncate(num_digits);
    let digit_exprs: Vec<Expr> =
      digits.iter().map(|&d| Expr::Integer(d)).collect();
    return Ok(Expr::List(
      vec![Expr::List(digit_exprs.into()), Expr::Integer(p + 1)].into(),
    ));
  }

  let decimal_exp = decimal_exp as i128;

  // Pad with zeros if we don't have enough digits
  while digits.len() < num_digits {
    digits.push(0);
  }

  // Truncate to requested number of digits
  digits.truncate(num_digits);

  let digit_exprs: Vec<Expr> =
    digits.iter().map(|&d| Expr::Integer(d)).collect();

  Ok(Expr::List(
    vec![Expr::List(digit_exprs.into()), Expr::Integer(decimal_exp)].into(),
  ))
}

/// FromDigits[list
pub fn from_digits_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "FromDigits expects 1 or 2 arguments".into(),
    ));
  }

  // The digit list is just a polynomial in the base, so *any* base works: an
  // integer takes the fast numeric path (a negative one giving the
  // "negabinary"/"negadecimal" representation, 1 the digit sum, 0 the last
  // digit alone), and anything else — a rational, a real, a symbol — goes
  // through the symbolic path, where `FromDigits[{1,2,3}, x]` is
  // `3 + 2*x + x^2`. There is nothing to reject.
  let numeric_base: Option<i128> = if args.len() == 2 {
    expr_to_i128(&args[1])
  } else {
    Some(10)
  };

  // FromDigits["XVII", "Roman"] reads a Roman numeral.
  if args.len() == 2
    && matches!(&args[0], Expr::String(_))
    && matches!(&args[1], Expr::String(b) if b == "Roman")
  {
    return from_roman_numeral_ast(&args[..1]);
  }

  // MixedRadix base: `FromDigits[{d0, ..., d_{n-1}}, MixedRadix[{r0, ..., r_{m-1}}]]`.
  // Each digit's weight is the running product of radices consumed from the
  // right (the last digit has weight 1, the next 1*r_{m-1}, then *r_{m-2}, …).
  // Leading digits with no radix left to consume drop out (weight 0). Digits
  // may be symbolic, in which case a Plus expression is returned.
  if args.len() == 2
    && let Expr::FunctionCall {
      name: mr,
      args: mr_args,
    } = &args[1]
    && mr == "MixedRadix"
    && mr_args.len() == 1
    && let Expr::List(radices) = &mr_args[0]
    && let Some(radix_vals) =
      radices.iter().map(expr_to_i128).collect::<Option<Vec<_>>>()
  {
    let Expr::List(digits) = &args[0] else {
      return Ok(unevaluated("FromDigits", args));
    };
    if digits.is_empty() {
      return Ok(Expr::Integer(0));
    }
    let n = digits.len();
    let m = radix_vals.len();
    // Build weights right-to-left; a digit with no remaining radix gets 0.
    let mut weights = vec![BigInt::from(0); n];
    weights[n - 1] = BigInt::from(1);
    let mut cur = BigInt::from(1);
    for j in 1..n {
      let ridx = m as isize - j as isize;
      if ridx < 0 {
        break;
      }
      cur *= BigInt::from(radix_vals[ridx as usize]);
      weights[n - 1 - j].clone_from(&cur);
    }
    let mut terms: Vec<Expr> = Vec::with_capacity(n);
    for (i, digit) in digits.iter().enumerate() {
      if weights[i] == BigInt::from(0) {
        continue;
      }
      if weights[i] == BigInt::from(1) {
        terms.push(digit.clone());
      } else {
        terms.push(times(vec![
          digit.clone(),
          bigint_to_expr(weights[i].clone()),
        ]));
      }
    }
    let sum = plus(terms);
    return crate::evaluator::evaluate_expr_to_expr(&sum);
  }

  // Symbolic base path: build an expanded polynomial sum in the base.
  // For `FromDigits[{d0, d1, ..., d_{n-1}}, b]` we produce
  //   d0*b^(n-1) + d1*b^(n-2) + ... + d_{n-2}*b + d_{n-1}.
  if numeric_base.is_none() {
    let items = match &args[0] {
      Expr::List(items) => items.clone(),
      _ => {
        return Ok(unevaluated("FromDigits", args));
      }
    };
    if items.is_empty() {
      return Ok(Expr::Integer(0));
    }
    let base_expr = args[1].clone();
    let n = items.len();
    let mut terms: Vec<Expr> = Vec::with_capacity(n);
    for (i, item) in items.iter().enumerate() {
      let power = (n - 1 - i) as i128;
      let term = if power == 0 {
        item.clone()
      } else {
        let base_pow = if power == 1 {
          base_expr.clone()
        } else {
          pow(base_expr.clone(), Expr::Integer(power))
        };
        times(vec![item.clone(), base_pow])
      };
      terms.push(term);
    }
    let sum = plus(terms);
    return crate::evaluator::evaluate_expr_to_expr(&sum);
  }

  let base: i128 = numeric_base.unwrap();
  let big_base = BigInt::from(base);

  // Handle string argument: FromDigits["1234"] or FromDigits["1a", 16]
  if let Expr::String(s) = &args[0] {
    let mut result = BigInt::from(0);
    for ch in s.chars() {
      let d = if ch.is_ascii_digit() {
        (ch as i128) - ('0' as i128)
      } else if ch.is_ascii_lowercase() {
        (ch as i128) - ('a' as i128) + 10
      } else if ch.is_ascii_uppercase() {
        (ch as i128) - ('A' as i128) + 10
      } else {
        return Ok(unevaluated("FromDigits", args));
      };
      // Wolfram allows digit values >= base (overflow digits)
      // e.g. FromDigits["a0"] in base 10 gives 10*10+0 = 100
      result = result * &big_base + BigInt::from(d);
    }
    return Ok(bigint_to_expr(result));
  }

  let Expr::List(items) = &args[0] else {
    return Ok(unevaluated("FromDigits", args));
  };

  // Handle {digit_list, {p1, p2, ...}} form: a list of decimal-point positions
  // threads FromDigits over each position, returning the list of results, e.g.
  // FromDigits[{{1, 0, 1}, {-1, 2}}] = {101/10000, 101/10}. Matches
  // wolframscript; an empty position list yields an empty list.
  if items.len() == 2
    && matches!(&items[0], Expr::List(_))
    && let Expr::List(positions) = &items[1]
  {
    let mut results = Vec::with_capacity(positions.len());
    for p in positions {
      let inner = Expr::List(vec![items[0].clone(), p.clone()].into());
      let mut call_args = vec![inner];
      if args.len() == 2 {
        call_args.push(args[1].clone());
      }
      results.push(from_digits_ast(&call_args)?);
    }
    return Ok(Expr::List(results.into()));
  }

  // {digits, e} with a recurring block (RealDigits of a rational): a final
  // sublist {r1, …, rk} repeats forever, adding
  // FromDigits[{r1, …, rk}] b^(e - m)/(b^k - 1) after the m plain digits.
  // A sublist anywhere else is just a (list-valued) digit, as in
  // wolframscript: FromDigits[{{{0}, 9, 1, 2, {0}}, 0}] is {57/625}.
  if items.len() == 2
    && let (Expr::List(digits), Some(exp)) =
      (&items[0], expr_to_i128(&items[1]))
    && digits.iter().any(|d| matches!(d, Expr::List(_)))
  {
    let b = Expr::Integer(base);
    let (plain, recurring) = match digits.last() {
      Some(Expr::List(block)) => (&digits[..digits.len() - 1], Some(block)),
      _ => (&digits[..], None),
    };
    let mut terms: Vec<Expr> = plain
      .iter()
      .enumerate()
      .map(|(i, d)| {
        times(vec![
          d.clone(),
          pow(b.clone(), Expr::Integer(exp - 1 - i as i128)),
        ])
      })
      .collect();
    if let Some(block) = recurring {
      let m = plain.len() as i128;
      let k = block.len() as i128;
      let value = from_digits_ast(&[Expr::List(block.clone()), b.clone()])?;
      if k > 0 {
        terms.push(times(vec![
          value,
          pow(b.clone(), Expr::Integer(exp - m)),
          pow(
            plus(vec![pow(b.clone(), Expr::Integer(k)), Expr::Integer(-1)]),
            Expr::Integer(-1),
          ),
        ]));
      }
    }
    return crate::evaluator::evaluate_expr_to_expr(&plus(terms));
  }

  // Handle {digit_list, exponent} form (output of RealDigits)
  // FromDigits[{{d1,d2,...,dn}, e}] = (d1*base^(e-1) + d2*base^(e-2) + ... + dn*base^(e-n))
  if items.len() == 2
    && let (Expr::List(digits), Some(exp)) =
      (&items[0], expr_to_i128(&items[1]))
    && digits.iter().all(|d| expr_to_i128(d).is_some())
  {
    let n = digits.len() as i128;
    // Compute the integer from digits
    let mut int_val = BigInt::from(0);
    for d in digits {
      let dv = expr_to_i128(d).unwrap();
      int_val = int_val * &big_base + BigInt::from(dv);
    }
    // The value is int_val * base^(exp - n)
    let shift = exp - n;
    if shift >= 0 {
      let factor = big_base.pow(shift as u32);
      return Ok(bigint_to_expr(int_val * factor));
    }
    // Need a rational: int_val / base^(-shift)
    let denom = big_base.pow((-shift) as u32);
    let (num, den) = rat_reduce_bigint(&int_val, &denom);
    if den == BigInt::from(1) {
      return Ok(bigint_to_expr(num));
    }
    return Ok(call(
      "Rational",
      vec![bigint_to_expr(num), bigint_to_expr(den)],
    ));
  }

  // Check if all items are numeric
  let all_numeric = items.iter().all(|item| expr_to_i128(item).is_some());

  if all_numeric {
    let mut result = BigInt::from(0);
    for item in items {
      let d = expr_to_i128(item).unwrap();
      result = result * &big_base + BigInt::from(d);
    }
    Ok(bigint_to_expr(result))
  } else {
    // Symbolic: build expression base*(base*(...) + d1) + d2
    // i.e., Horner form: ((d0 * base + d1) * base + d2) * base + ...
    let base_expr = Expr::Integer(base);
    let mut result = items[0].clone();
    for item in &items[1..] {
      // result = result * base + item
      result = plus(vec![times(vec![base_expr.clone(), result]), item.clone()]);
      result = crate::evaluator::evaluate_expr_to_expr(&result)?;
    }
    Ok(result)
  }
}

/// IntegerLength[n] - Number of digits of n in base 10
/// IntegerLength[n, b] - Number of digits in base b
/// IntegerLength[12345] => 5
/// IntegerLength[255, 16] => 2
/// IntegerLength[0] => 0
pub fn integer_length_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "IntegerLength expects 1 or 2 arguments".into(),
    ));
  }

  // IntegerLength[n, MixedRadix[{...}]] is the number of mixed-radix digits,
  // i.e. Length[IntegerDigits[n, MixedRadix[{...}]]]. Delegate to IntegerDigits
  // so the leading-zero stripping and radix handling stay in one place.
  if args.len() == 2
    && matches!(&args[1], Expr::FunctionCall { name, .. } if name == "MixedRadix")
  {
    let digits = integer_digits_ast(&[args[0].clone(), args[1].clone()])?;
    if let Expr::List(items) = &digits {
      return Ok(Expr::Integer(items.len() as i128));
    }
    // IntegerDigits left it unevaluated (e.g. bad radix) — mirror that.
    return Ok(unevaluated("IntegerLength", args));
  }

  let base_i128 = if args.len() == 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => b,
      Some(b) => {
        crate::emit_message(&format!(
          "IntegerLength::ibase: Base {b} is not an integer greater than 1."
        ));
        return Ok(unevaluated("IntegerLength", args));
      }
      None => {
        return Ok(unevaluated("IntegerLength", args));
      }
    }
  } else {
    10
  };

  // Try BigInt path first (handles both Integer and BigInteger)
  if let Some(n) = expr_to_bigint(&args[0]) {
    if n.is_zero() {
      return Ok(Expr::Integer(0));
    }
    let base_big = BigInt::from(base_i128);
    let mut abs_n = if n < BigInt::zero() { -n } else { n };
    let mut count = 0i128;
    while abs_n > BigInt::zero() {
      abs_n /= &base_big;
      count += 1;
    }
    return Ok(Expr::Integer(count));
  }

  Ok(unevaluated("IntegerLength", args))
}

/// IntegerReverse[n] - reverse the digits of an integer in base 10.
/// IntegerReverse[n, b] - reverse the digits of n in base b.
/// IntegerReverse[n, b, len] - reverse exactly the `len` least-significant
/// base-b digits of n (padding with leading zeros), matching
/// FromDigits[Reverse[IntegerDigits[n, b, len]], b].
pub fn integer_reverse_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "IntegerReverse expects 1, 2, or 3 arguments".into(),
    ));
  }

  let uneval = || Ok(unevaluated("IntegerReverse", args));

  // A mixed-radix base: the digits in that radix, reversed and read back.
  if args.len() == 2
    && matches!(&args[1], Expr::FunctionCall { name, .. } if name == "MixedRadix")
    && matches!(&args[0], Expr::Integer(_) | Expr::BigInteger(_))
  {
    use crate::evaluator::evaluate_function_call_ast as ev;
    let digits = ev("IntegerDigits", args)?;
    let Expr::List(ds) = &digits else {
      return uneval();
    };
    let reversed: Vec<Expr> = ds.iter().rev().cloned().collect();
    return ev("FromDigits", &[Expr::List(reversed.into()), args[1].clone()]);
  }

  let base = if args.len() >= 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => b,
      Some(_) => {
        // wolframscript says "at position 2 of", not "in".
        crate::emit_message(&format!(
          "IntegerReverse::ibmr: Positive integer greater than 1 or mixed \
           radix specification expected at position 2 of {}.",
          crate::syntax::format_expr(
            &unevaluated("IntegerReverse", args),
            crate::syntax::ExprForm::Output
          )
        ));
        return uneval();
      }
      None => return uneval(),
    }
  } else {
    10
  };

  // Optional digit-count: reverse exactly `len` low-order digits.
  let len = if args.len() == 3 {
    match expr_to_i128(&args[2]) {
      Some(l) if l >= 0 => Some(l),
      Some(_) => {
        crate::emit_message(&format!(
          "IntegerReverse::intpm: Positive machine-sized integer expected at position 3 in {}.",
          expr_to_string(&unevaluated("IntegerReverse", args))
        ));
        return uneval();
      }
      None => return uneval(),
    }
  } else {
    None
  };

  // Handle BigInteger
  if let Some(n) = expr_to_bigint(&args[0]) {
    let mut abs_n = if n < BigInt::zero() { -n } else { n };
    let base_big = BigInt::from(base);
    let mut result = BigInt::zero();
    match len {
      // Fixed digit count: take exactly `len` low-order digits, most
      // significant last, so leading zeros participate in the reversal.
      Some(len) => {
        for _ in 0..len {
          result = result * &base_big + (&abs_n % &base_big);
          abs_n /= &base_big;
        }
      }
      // Reverse all significant digits.
      None => {
        while abs_n > BigInt::zero() {
          result = result * &base_big + (&abs_n % &base_big);
          abs_n /= &base_big;
        }
      }
    }
    return Ok(bigint_to_expr(result));
  }

  uneval()
}

const ONES: [&str; 20] = [
  "zero",
  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
  "ten",
  "eleven",
  "twelve",
  "thirteen",
  "fourteen",
  "fifteen",
  "sixteen",
  "seventeen",
  "eighteen",
  "nineteen",
];

const TENS: [&str; 10] = [
  "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
  "ninety",
];

const SCALES: [&str; 7] = [
  "",
  "thousand",
  "million",
  "billion",
  "trillion",
  "quadrillion",
  "quintillion",
];

/// Spell out a number 0..=999 in English words.
/// Uses U+2010 HYPHEN for compound numbers like "twenty‐one".
fn spell_below_1000(n: u64) -> String {
  if n == 0 {
    return String::new();
  }
  let mut parts = Vec::new();
  let hundreds = n / 100;
  let remainder = n % 100;
  if hundreds > 0 {
    parts.push(format!("{} hundred", ONES[hundreds as usize]));
  }
  if remainder > 0 {
    if remainder < 20 {
      parts.push(ONES[remainder as usize].to_string());
    } else {
      let tens = remainder / 10;
      let ones = remainder % 10;
      if ones == 0 {
        parts.push(TENS[tens as usize].to_string());
      } else {
        // U+2010 HYPHEN between tens and ones
        parts.push(format!(
          "{}\u{2010}{}",
          TENS[tens as usize], ONES[ones as usize]
        ));
      }
    }
  }
  parts.join(" ")
}

/// Convert a cardinal English number name to its ordinal form. Only the last
/// word changes, and the tens-ones hyphen becomes a plain `-`:
/// "forty‐two" -> "forty-second", "one hundred" -> "one hundredth".
fn cardinal_to_ordinal(cardinal: &str) -> String {
  let normalized = cardinal.replace('\u{2010}', "-");
  match normalized.rfind([' ', '-']) {
    Some(i) => {
      let (prefix, last) = normalized.split_at(i + 1);
      format!("{}{}", prefix, ordinalize_word(last))
    }
    None => ordinalize_word(&normalized),
  }
}

/// Ordinal form of a single number word.
fn ordinalize_word(w: &str) -> String {
  match w {
    "one" => "first".to_string(),
    "two" => "second".to_string(),
    "three" => "third".to_string(),
    "five" => "fifth".to_string(),
    "eight" => "eighth".to_string(),
    "nine" => "ninth".to_string(),
    "twelve" => "twelfth".to_string(),
    "twenty" => "twentieth".to_string(),
    "thirty" => "thirtieth".to_string(),
    "forty" => "fortieth".to_string(),
    "fifty" => "fiftieth".to_string(),
    "sixty" => "sixtieth".to_string(),
    "seventy" => "seventieth".to_string(),
    "eighty" => "eightieth".to_string(),
    "ninety" => "ninetieth".to_string(),
    _ => format!("{w}th"),
  }
}

/// The cardinal English name of an integer, or None when it can't be named
/// (e.g. magnitude beyond the known scale words).
fn integer_name_cardinal(n: i128) -> Option<String> {
  let negative = n < 0;
  let abs_n = n.unsigned_abs();

  if abs_n == 0 {
    return Some("zero".to_string());
  }

  // For numbers 1..=999, spell out entirely in words
  if abs_n <= 999 {
    let word = spell_below_1000(abs_n as u64);
    return Some(if negative {
      format!("negative {word}")
    } else {
      word
    });
  }

  // For numbers >= 1000, break into groups of 3 digits.
  // Higher groups use digit representation, the lowest group (< 1000) uses words.
  let mut groups: Vec<(u64, usize)> = Vec::new(); // (group_value, scale_index)
  let mut remaining = abs_n as u64;
  let mut scale_idx = 0;
  while remaining > 0 {
    let group = remaining % 1000;
    if group > 0 {
      groups.push((group, scale_idx));
    }
    remaining /= 1000;
    scale_idx += 1;
  }
  groups.reverse();

  let mut parts = Vec::new();
  for &(group, sidx) in &groups {
    if sidx == 0 {
      // Lowest group: use digits (for numbers >= 1000)
      parts.push(format!("{group}"));
    } else {
      // Higher groups: use digits + scale word
      let scale = if sidx < SCALES.len() {
        SCALES[sidx]
      } else {
        return None;
      };
      parts.push(format!("{group} {scale}"));
    }
  }

  let result = parts.join(" ");
  Some(if negative {
    format!("negative {result}")
  } else {
    result
  })
}

/// The fully-spelled-out "Words" form of an integer name. Unlike the default
/// cardinal form (which uses digit groups for values >= 1000, e.g.
/// `1 million 234 thousand 567`), this spells every group in words and joins
/// the non-zero groups with ", ":
/// `IntegerName[1234567, "Words"]` =
///   "one million, two hundred thirty‐four thousand, five hundred sixty‐seven".
fn integer_name_words(n: i128) -> Option<String> {
  let negative = n < 0;
  let abs_n = n.unsigned_abs();
  if abs_n == 0 {
    return Some("zero".to_string());
  }

  // Break into groups of three digits, low to high.
  let mut groups: Vec<(u64, usize)> = Vec::new();
  let mut remaining = abs_n;
  let mut scale_idx = 0;
  while remaining > 0 {
    let group = (remaining % 1000) as u64;
    if group > 0 {
      groups.push((group, scale_idx));
    }
    remaining /= 1000;
    scale_idx += 1;
  }
  groups.reverse();

  let mut parts = Vec::with_capacity(groups.len());
  for &(group, sidx) in &groups {
    let words = spell_below_1000(group);
    if sidx == 0 {
      parts.push(words);
    } else {
      let scale = if sidx < SCALES.len() {
        SCALES[sidx]
      } else {
        return None;
      };
      parts.push(format!("{words} {scale}"));
    }
  }

  let result = parts.join(", ");
  Some(if negative {
    format!("negative {result}")
  } else {
    result
  })
}

/// German word for the units digit 1-9. `eins_ok` chooses the standalone
/// "eins" over the compound stem "ein" (used before "und"/"hundert"/"tausend").
fn german_unit(d: u8, eins_ok: bool) -> &'static str {
  match d {
    1 => {
      if eins_ok {
        "eins"
      } else {
        "ein"
      }
    }
    2 => "zwei",
    3 => "drei",
    4 => "vier",
    5 => "fünf",
    6 => "sechs",
    7 => "sieben",
    8 => "acht",
    9 => "neun",
    _ => "",
  }
}

/// German word for 10-19 (each an atomic morpheme).
fn german_teen(r: u8) -> &'static str {
  match r {
    10 => "zehn",
    11 => "elf",
    12 => "zwölf",
    13 => "dreizehn",
    14 => "vierzehn",
    15 => "fünfzehn",
    16 => "sechzehn",
    17 => "siebzehn",
    18 => "achtzehn",
    19 => "neunzehn",
    _ => "",
  }
}

/// German word for a multiple of ten 20-90 (atomic morpheme).
fn german_ten(t: u8) -> &'static str {
  match t {
    20 => "zwanzig",
    30 => "dreißig",
    40 => "vierzig",
    50 => "fünfzig",
    60 => "sechzig",
    70 => "siebzig",
    80 => "achtzig",
    90 => "neunzig",
    _ => "",
  }
}

/// Push the morphemes for 1..=99 onto `out`. `eins_ok` lets a trailing
/// standalone 1 spell as "eins" (otherwise "ein").
fn german_below_100(r: u8, eins_ok: bool, out: &mut Vec<String>) {
  if r == 0 {
    return;
  }
  if r <= 9 {
    out.push(german_unit(r, eins_ok).to_string());
  } else if r <= 19 {
    out.push(german_teen(r).to_string());
  } else if r.is_multiple_of(10) {
    out.push(german_ten(r).to_string());
  } else {
    // e.g. 21 → ein­und­zwanzig (unit before "und" is always "ein").
    let u = r % 10;
    out.push(german_unit(u, false).to_string());
    out.push("und".to_string());
    out.push(german_ten(r - u).to_string());
  }
}

/// Push the morphemes for a 1..=999 group onto `out`.
fn german_group(g: u16, eins_ok: bool, out: &mut Vec<String>) {
  let h = (g / 100) as u8;
  let r = (g % 100) as u8;
  if h > 0 {
    out.push(german_unit(h, false).to_string());
    out.push("hundert".to_string());
  }
  german_below_100(r, eins_ok, out);
}

/// German cardinal morphemes for 0 < n < 1_000_000.
fn german_cardinal_morphemes(n: u32) -> Option<Vec<String>> {
  if n == 0 || n >= 1_000_000 {
    return None;
  }
  let mut out = Vec::new();
  let th = (n / 1000) as u16;
  let un = (n % 1000) as u16;
  if th > 0 {
    german_group(th, false, &mut out);
    out.push("tausend".to_string());
  }
  if un > 0 {
    german_group(un, true, &mut out);
  }
  Some(out)
}

/// Transform the final cardinal morpheme into its ordinal form. Units 1, 3, 7,
/// 8 are irregular (erste/dritte/siebte/achte); 1-19 take "-te", multiples of
/// ten and hundert/tausend take "-ste".
fn german_ordinal_suffix(word: &str) -> String {
  match word {
    "eins" | "ein" => "erste".to_string(),
    "zwei" => "zweite".to_string(),
    "drei" => "dritte".to_string(),
    "vier" => "vierte".to_string(),
    "fünf" => "fünfte".to_string(),
    "sechs" => "sechste".to_string(),
    "sieben" => "siebte".to_string(),
    "acht" => "achte".to_string(),
    "neun" => "neunte".to_string(),
    "zehn" => "zehnte".to_string(),
    "elf" => "elfte".to_string(),
    "zwölf" => "zwölfte".to_string(),
    "zwanzig" | "dreißig" | "vierzig" | "fünfzig" | "sechzig" | "siebzig"
    | "achtzig" | "neunzig" | "hundert" | "tausend" => format!("{word}ste"),
    // Teens (dreizehn..neunzehn).
    _ => format!("{word}te"),
  }
}

/// German long-scale group word (singular, plural) for the 3-digit group at
/// index 2..=5 (10^6, 10^9, 10^12, 10^15). German uses the long scale, so the
/// scale name alternates -illion / -illiarde every group. Index >= 6 (>=10^18)
/// is beyond the range wolframscript spells.
fn german_scale_word(group_idx: usize) -> Option<(&'static str, &'static str)> {
  match group_idx {
    2 => Some(("Million", "Millionen")),
    3 => Some(("Milliarde", "Milliarden")),
    4 => Some(("Billion", "Billionen")),
    5 => Some(("Billiarde", "Billiarden")),
    _ => None,
  }
}

/// True for the German long-scale nouns, which take the "-ste" ordinal suffix.
fn is_german_scale_word(w: &str) -> bool {
  matches!(
    w,
    "Million"
      | "Millionen"
      | "Milliarde"
      | "Milliarden"
      | "Billion"
      | "Billionen"
      | "Billiarde"
      | "Billiarden"
  )
}

/// Spell the multiplier 1..=999 of a (feminine) long-scale noun, soft-hyphen
/// joined. A trailing standalone "eins" becomes feminine "eine" to agree with
/// the noun, e.g. 1 -> "eine", 101 -> "ein­hundert­eine".
fn german_scale_count(g: u16) -> String {
  let mut morphemes = Vec::new();
  german_group(g, true, &mut morphemes);
  if let Some(last) = morphemes.last_mut()
    && last == "eins"
  {
    *last = "eine".to_string();
  }
  morphemes.join("\u{00AD}")
}

/// Spell `n` in [10^6, 10^18) using the German long scale. Scale groups read
/// "<multiplier> <ScaleNoun>" (singular noun only when the multiplier is 1) and
/// are space-joined; the sub-million remainder is appended as one soft-hyphen
/// joined token. Ordinal suffixes the last morpheme of the last token.
fn german_millions(n: u64, ordinal: bool) -> Option<String> {
  let mut groups = [0u16; 6];
  let mut rem = n;
  for slot in &mut groups {
    *slot = (rem % 1000) as u16;
    rem /= 1000;
  }
  let mut tokens: Vec<String> = Vec::new();
  for idx in (2..=5).rev() {
    let g = groups[idx];
    if g == 0 {
      continue;
    }
    let (sing, plur) = german_scale_word(idx)?;
    let scale = if g == 1 { sing } else { plur };
    tokens.push(format!("{} {}", german_scale_count(g), scale));
  }
  let below = (groups[1] as u32) * 1000 + groups[0] as u32;
  if below > 0 {
    tokens.push(german_cardinal_morphemes(below)?.join("\u{00AD}"));
  }
  let mut out = tokens.join(" ");
  if ordinal {
    // The ordinal suffix attaches to the last morpheme, which is delimited by a
    // space or soft-hyphen; long-scale nouns take "-ste", others the usual form.
    let split = out
      .rfind([' ', '\u{00AD}'])
      .map_or(0, |i| i + out[i..].chars().next().unwrap().len_utf8());
    let last = out[split..].to_string();
    let suffixed = if is_german_scale_word(&last) {
      format!("{last}ste")
    } else {
      german_ordinal_suffix(&last)
    };
    out.truncate(split);
    out.push_str(&suffixed);
  }
  Some(out)
}

/// Spell `n` in German as a cardinal or ordinal. Morphemes are joined by the
/// soft-hyphen U+00AD like wolframscript; negatives get a "minus " prefix.
/// Returns None for |n| >= 10^18, which is beyond wolframscript's German range.
fn integer_name_german(n: i128, ordinal: bool) -> Option<String> {
  let negative = n < 0;
  let abs = n.unsigned_abs();
  if abs >= 1_000_000_000_000_000_000 {
    return None;
  }
  let body = if abs == 0 {
    if ordinal { "nullte" } else { "null" }.to_string()
  } else if abs < 1_000_000 {
    let mut morphemes = german_cardinal_morphemes(abs as u32)?;
    if ordinal {
      let last = morphemes.len() - 1;
      morphemes[last] = german_ordinal_suffix(&morphemes[last]);
    }
    morphemes.join("\u{00AD}")
  } else {
    german_millions(abs as u64, ordinal)?
  };
  Some(if negative {
    format!("minus {body}")
  } else {
    body
  })
}

pub fn integer_name_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("IntegerName", args));

  // IntegerName works on lists; thread over the first argument, carrying any
  // second argument (the name type) along.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| {
        let mut new_args = vec![item.clone()];
        if args.len() == 2 {
          new_args.push(args[1].clone());
        }
        integer_name_ast(&new_args)
      })
      .collect();
    return Ok(Expr::List(results?.into()));
  }

  let Some(n) = expr_to_i128(&args[0]) else {
    return unevaluated();
  };

  let Some(cardinal) = integer_name_cardinal(n) else {
    return unevaluated();
  };

  // The qualifier may be a plain string ("Ordinal"/"Words") or a list such as
  // {"English", "Ordinal"} / {"Ordinal"} / {"English", "Words"}. Collect the
  // string tokens and pick out the requested form; an unrecognized token
  // (e.g. a non-English language Woxi cannot spell) leaves it unevaluated.
  if let Some(qualifier) = args.get(1) {
    let tokens: Vec<&str> = match qualifier {
      Expr::String(s) => vec![s.as_str()],
      Expr::List(items) => {
        let mut t = Vec::with_capacity(items.len());
        for it in items {
          match it {
            Expr::String(s) => t.push(s.as_str()),
            _ => return unevaluated(),
          }
        }
        t
      }
      _ => return unevaluated(),
    };
    let mut want_ordinal = false;
    let mut want_words = false;
    let mut german = false;
    for tok in &tokens {
      match *tok {
        "Ordinal" => want_ordinal = true,
        "Words" => want_words = true,
        "German" => german = true,
        // English and German are the spellable languages; "Cardinal" is the
        // default. Anything else is unsupported.
        "English" | "Cardinal" => {}
        _ => return unevaluated(),
      }
    }
    if german {
      // German "Words" form is not supported; only cardinal and ordinal.
      if want_words {
        return unevaluated();
      }
      return if let Some(s) = integer_name_german(n, want_ordinal) {
        Ok(Expr::String(s))
      } else {
        // wolframscript spells German only below 10^18; larger values warn.
        if n.unsigned_abs() >= 1_000_000_000_000_000_000 {
          crate::emit_message(&format!(
            "IntegerName::outrange: Number {n} is too large for the \
             algorithms available for German."
          ));
        }
        unevaluated()
      };
    }
    if want_ordinal {
      return Ok(Expr::String(cardinal_to_ordinal(&cardinal)));
    }
    if want_words && let Some(words) = integer_name_words(n) {
      return Ok(Expr::String(words));
    }
  }
  Ok(Expr::String(cardinal))
}

pub fn roman_numeral_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // RomanNumeral[n] - convert integer to Roman numeral string
  // Returns a Symbol (not a String) — e.g. RomanNumeral[2025] => MMXXV
  // Negative integers: convert the absolute value
  // Zero: returns N
  // Non-integer: return unevaluated

  let Some(n) = expr_to_i128(&args[0]) else {
    return Ok(unevaluated("RomanNumeral", args));
  };

  if n == 0 {
    return Ok(str_expr("N"));
  }

  let abs_n = n.unsigned_abs() as u64;

  // For values >= 5000, Wolfram uses display forms with overscript bars.
  // We support up to 4999 with plain Roman numerals.
  if abs_n >= 5000 {
    return Ok(unevaluated("RomanNumeral", args));
  }

  const VALUES: [(u64, &str); 13] = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
  ];

  let mut result = String::new();
  let mut remaining = abs_n;
  for &(value, numeral) in &VALUES {
    while remaining >= value {
      result.push_str(numeral);
      remaining -= value;
    }
  }

  Ok(Expr::String(result))
}

/// Convergents[{a0, a1, a2, ...}] - list of convergents of a continued fraction
/// Convergents[x, n] - convergents of the continued fraction of x, up to n terms
pub fn convergents_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // If first arg is a number (not a list), compute ContinuedFraction first
  let cf_list = match &args[0] {
    Expr::List(_) => args[0].clone(),
    _ => continued_fraction_ast(args)?,
  };

  let Expr::List(elements) = &cf_list else {
    return Ok(unevaluated("Convergents", args));
  };

  if elements.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  // Collect all integers
  let mut ints: Vec<num_bigint::BigInt> = Vec::new();
  for elem in elements {
    match elem {
      Expr::Integer(n) => ints.push(num_bigint::BigInt::from(*n)),
      Expr::BigInteger(n) => ints.push(n.clone()),
      _ => {
        return Ok(unevaluated("Convergents", args));
      }
    }
  }

  // The recurrence h[n] = a[n] h[n-1] + h[n-2], k[n] = a[n] k[n-1] + k[n-2]
  // (h[-1] = 1, k[-1] = 0), in big integers: a long expansion outgrows i128.
  let to_expr = |h: &num_bigint::BigInt, k: &num_bigint::BigInt| {
    crate::evaluator::evaluate_function_call_ast(
      "Rational",
      &[
        crate::functions::math_ast::bigint_to_expr(h.clone()),
        crate::functions::math_ast::bigint_to_expr(k.clone()),
      ],
    )
  };
  let mut result = Vec::with_capacity(ints.len());
  let (mut h_prev2, mut h_prev1) =
    (num_bigint::BigInt::from(1), ints[0].clone());
  let (mut k_prev2, mut k_prev1) =
    (num_bigint::BigInt::from(0), num_bigint::BigInt::from(1));
  result.push(to_expr(&h_prev1, &k_prev1)?);
  for a in &ints[1..] {
    let h = a * &h_prev1 + &h_prev2;
    let k = a * &k_prev1 + &k_prev2;
    result.push(to_expr(&h, &k)?);
    h_prev2 = std::mem::replace(&mut h_prev1, h);
    k_prev2 = std::mem::replace(&mut k_prev1, k);
  }

  Ok(Expr::List(result.into()))
}

fn make_rational_expr(num: i128, den: i128) -> Expr {
  if den == 1 {
    Expr::Integer(num)
  } else if den == -1 {
    Expr::Integer(-num)
  } else {
    let (n, d) = rat_reduce(num, den);
    if d < 0 {
      if -d == 1 {
        Expr::Integer(-n)
      } else {
        div2(Expr::Integer(-n), Expr::Integer(-d))
      }
    } else if d == 1 {
      Expr::Integer(n)
    } else {
      div2(Expr::Integer(n), Expr::Integer(d))
    }
  }
}

/// NumberDigit[x, n] — returns the digit at position n of a real number x.
/// Position 0 is the ones digit, positive n goes left (tens, hundreds, ...),
/// negative n goes right (tenths, hundredths, ...).
pub fn number_digit_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use num_bigint::BigUint;
  use num_traits::{One, ToPrimitive};
  let unevaluated = || unevaluated("NumberDigit", args);
  if args.len() != 2 && args.len() != 3 {
    return Ok(unevaluated());
  }
  // Base (default 10); integers below 2 emit ::rbase
  let base: u32 = if args.len() == 3 {
    match &args[2] {
      Expr::Integer(b) if *b >= 2 && *b <= 1_000_000 => *b as u32,
      Expr::Integer(b) => {
        crate::emit_message(&format!(
          "NumberDigit::rbase: Base {b} is not a real number greater than 1."
        ));
        return Ok(unevaluated());
      }
      _ => return Ok(unevaluated()),
    }
  } else {
    10
  };
  // Position: an integer or a list of integers
  let positions: Vec<i128> = match &args[1] {
    Expr::Integer(k) => vec![*k],
    Expr::List(items)
      if items.iter().all(|i| matches!(i, Expr::Integer(_))) =>
    {
      items
        .iter()
        .map(|i| match i {
          Expr::Integer(k) => *k,
          _ => unreachable!(),
        })
        .collect()
    }
    other => {
      crate::emit_message(&format!(
        "NumberDigit::badspec: Argument {} at position 2 should be an integer or a list of integers.",
        crate::syntax::format_expr(other, crate::syntax::ExprForm::Output)
      ));
      return Ok(unevaluated());
    }
  };
  if positions.iter().any(|k| k.unsigned_abs() > 10_000) {
    return Ok(unevaluated());
  }

  // Digit of the exact non-negative rational p/q at base^k:
  // floor(p * base^max(0,-k) / (q * base^max(0,k))) mod base
  let exact_digit = |p: &BigUint, q: &BigUint, k: i128| -> i128 {
    let bb = BigUint::from(base);
    let (num, den) = if k >= 0 {
      (p.clone(), q * bb.pow(k as u32))
    } else {
      (p * bb.pow((-k) as u32), q.clone())
    };
    ((num / den) % bb).to_i128().unwrap_or(0)
  };

  enum Value {
    Exact(BigUint, BigUint),
    // Machine real: exact binary rational plus the ulp exponent (digits
    // finer than one ulp are Indeterminate, matching wolframscript)
    Machine(BigUint, BigUint, i64),
    // Machine real in base 10: shortest round-trip decimal digits, the
    // exponent of the leading digit, and the ulp exponent
    Shortest(Vec<u8>, i64, i64),
    Zero,
    // Pre-extracted radix digits (symbolic numeric constants): value =
    // 0.digits * base^exponent
    Radix(Vec<u8>, i64),
  }
  let max_abs_k = positions.iter().map(|k| k.unsigned_abs()).max().unwrap();
  let value = match &args[0] {
    Expr::Integer(n) => {
      Value::Exact(BigUint::from(n.unsigned_abs()), BigUint::one())
    }
    Expr::BigInteger(n) => Value::Exact(n.magnitude().clone(), BigUint::one()),
    Expr::FunctionCall { name, args: ra }
      if name == "Rational" && ra.len() == 2 =>
    {
      match (&ra[0], &ra[1]) {
        (Expr::Integer(p), Expr::Integer(q)) if *q != 0 => Value::Exact(
          BigUint::from(p.unsigned_abs()),
          BigUint::from(q.unsigned_abs()),
        ),
        _ => return Ok(unevaluated()),
      }
    }
    Expr::Real(f) => {
      let a = f.abs();
      if a == 0.0 {
        Value::Zero
      } else if !a.is_finite() {
        return Ok(unevaluated());
      } else if base == 10 {
        // wolframscript reads the digits off the shortest round-trip
        // decimal representation (zeros beyond it), with digits finer
        // than one ulp Indeterminate
        let bits = a.to_bits();
        let raw_exp = ((bits >> 52) & 0x7ff) as i64;
        let e2 = if raw_exp == 0 { -1074 } else { raw_exp - 1075 };
        // Shortest representation, normalized to digits + top exponent
        let s = format!("{a:e}"); // like 2.10345e2
        let (mant, exp10) = s.split_once('e').unwrap();
        let exp10: i64 = exp10.parse().unwrap();
        let digits: Vec<u8> = mant
          .bytes()
          .filter(u8::is_ascii_digit)
          .map(|b| b - b'0')
          .collect();
        Value::Shortest(digits, exp10, e2)
      } else {
        // Non-decimal bases use the exact binary rational
        let bits = a.to_bits();
        let raw_exp = ((bits >> 52) & 0x7ff) as i64;
        let mantissa = if raw_exp == 0 {
          bits & 0xf_ffff_ffff_ffff
        } else {
          (bits & 0xf_ffff_ffff_ffff) | (1u64 << 52)
        };
        let e2 = if raw_exp == 0 { -1074 } else { raw_exp - 1075 };
        let (p, q) = if e2 >= 0 {
          (BigUint::from(mantissa) << (e2 as u64), BigUint::one())
        } else {
          (BigUint::from(mantissa), BigUint::one() << ((-e2) as u64))
        };
        Value::Machine(p, q, e2)
      }
    }
    other => {
      // Symbolic numerics (Pi, E, Sqrt[2], ...) via arbitrary precision;
      // digit extraction uses radix conversion (base 2, 8, 10, 16 only)
      let radix = match base {
        2 => Some(astro_float::Radix::Bin),
        8 => Some(astro_float::Radix::Oct),
        10 => Some(astro_float::Radix::Dec),
        16 => Some(astro_float::Radix::Hex),
        _ => None,
      };
      let bits = (((max_abs_k as f64) + 60.0) * (base as f64).log2()).ceil()
        as usize
        + 64;
      let mut cc = astro_float::Consts::new()
        .map_err(|e| InterpreterError::EvaluationError(format!("N: {e}")))?;
      let rm = astro_float::RoundingMode::ToEven;
      if let Ok(bf) = expr_to_bigfloat(other, bits, rm, &mut cc) {
        let Some(radix) = radix else {
          return Ok(unevaluated());
        };
        if bf.is_zero() {
          Value::Zero
        } else {
          match bf.convert_to_radix(radix, rm, &mut cc) {
            Ok((_, digits, exponent)) => Value::Radix(digits, exponent as i64),
            Err(_) => return Ok(unevaluated()),
          }
        }
      } else {
        crate::emit_message(&format!(
          "NumberDigit::num: Argument {} should be a number.",
          crate::syntax::format_expr(&args[0], crate::syntax::ExprForm::Output)
        ));
        return Ok(unevaluated());
      }
    }
  };

  let digit_at = |k: i128| -> Expr {
    match &value {
      Value::Zero => Expr::Integer(0),
      Value::Exact(p, q) => Expr::Integer(exact_digit(p, q, k)),
      Value::Machine(p, q, e2) => {
        // Indeterminate when base^k is finer than one ulp (2^e2):
        // exact comparison base^k < 2^e2
        let bb = BigUint::from(base);
        let (lhs, rhs) = if k >= 0 {
          if *e2 >= 0 {
            (bb.pow(k as u32), BigUint::one() << (*e2 as u64))
          } else {
            (bb.pow(k as u32) << ((-e2) as u64), BigUint::one())
          }
        } else if *e2 >= 0 {
          (
            BigUint::one(),
            (BigUint::one() << (*e2 as u64)) * bb.pow((-k) as u32),
          )
        } else {
          (BigUint::one() << ((-e2) as u64), bb.pow((-k) as u32))
        };
        if lhs < rhs {
          id_expr("Indeterminate")
        } else {
          Expr::Integer(exact_digit(p, q, k))
        }
      }
      Value::Shortest(digits, exp10, e2) => {
        // Indeterminate when 10^k is finer than one ulp (2^e2)
        let bb = BigUint::from(10u32);
        let (lhs, rhs) = if k >= 0 {
          if *e2 >= 0 {
            (bb.pow(k as u32), BigUint::one() << (*e2 as u64))
          } else {
            (bb.pow(k as u32) << ((-e2) as u64), BigUint::one())
          }
        } else if *e2 >= 0 {
          (
            BigUint::one(),
            (BigUint::one() << (*e2 as u64)) * bb.pow((-k) as u32),
          )
        } else {
          (BigUint::one() << ((-e2) as u64), bb.pow((-k) as u32))
        };
        if lhs < rhs {
          return id_expr("Indeterminate");
        }
        let idx = *exp10 as i128 - k;
        if idx < 0 {
          Expr::Integer(0)
        } else if (idx as usize) < digits.len() {
          Expr::Integer(digits[idx as usize] as i128)
        } else {
          Expr::Integer(0)
        }
      }
      Value::Radix(digits, exponent) => {
        let idx = *exponent as i128 - 1 - k;
        if idx < 0 {
          Expr::Integer(0)
        } else if (idx as usize) < digits.len() {
          Expr::Integer(digits[idx as usize] as i128)
        } else {
          Expr::Integer(0)
        }
      }
    }
  };

  if let Expr::List(_) = &args[1] {
    Ok(Expr::List(
      positions
        .iter()
        .map(|&k| digit_at(k))
        .collect::<Vec<_>>()
        .into(),
    ))
  } else {
    Ok(digit_at(positions[0]))
  }
}

/// NumberExpand[n] / NumberExpand[n, b] - place-value decomposition of
/// an exact integer (zeros kept, each term carrying the sign of n);
/// rationals come back as a one-element list, matching wolframscript.
/// An integer base below 2 emits NumberExpand::rbase; symbolic input
/// and machine reals stay unevaluated (the real-number expansion with
/// its trailing machine-precision residual term is not implemented).
pub fn number_expand_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated("NumberExpand", args);
  if args.is_empty() || args.len() > 3 {
    return Ok(unevaluated(args));
  }
  // Base validation happens before the value is looked at
  let base: i128 = if args.len() >= 2 {
    match &args[1] {
      Expr::Integer(b) if *b >= 2 => *b,
      Expr::Integer(b) => {
        crate::emit_message(&format!(
          "NumberExpand::rbase: Base {b} is not a real number greater than 1."
        ));
        return Ok(unevaluated(args));
      }
      _ => return Ok(unevaluated(args)),
    }
  } else {
    10
  };
  // Optional length: the result has exactly `len` elements. When `len` exceeds
  // the digit count the expansion is padded with trailing zeros; when it is
  // smaller, the low-order terms are summed into the final element.
  let len: Option<usize> = if args.len() == 3 {
    match &args[2] {
      Expr::Integer(l) if *l >= 1 => Some(*l as usize),
      _ => return Ok(unevaluated(args)),
    }
  } else {
    None
  };

  match &args[0] {
    Expr::Integer(n) => {
      let sign = n.signum();
      let mut digits: Vec<i128> = Vec::new();
      let mut num = n.unsigned_abs();
      let base_u = base as u128;
      while num != 0 {
        digits.push((num % base_u) as i128);
        num /= base_u;
      }
      // digits[i] sits at place base^i; collect the place-value terms with the
      // highest place first.
      let mut terms: Vec<i128> = digits
        .iter()
        .enumerate()
        .rev()
        .map(|(place, d)| sign * d * base.pow(place as u32))
        .collect();
      if terms.is_empty() {
        terms.push(0); // n == 0
      }
      if let Some(len) = len {
        if len >= terms.len() {
          terms.resize(len, 0); // pad with trailing zeros
        } else {
          // Keep the top len-1 terms; fold the remainder into the last.
          let tail: i128 = terms[len - 1..].iter().sum();
          terms.truncate(len - 1);
          terms.push(tail);
        }
      }
      Ok(Expr::List(terms.into_iter().map(Expr::Integer).collect()))
    }
    Expr::FunctionCall { name, .. } if name == "Rational" => {
      Ok(Expr::List(vec![args[0].clone()].into()))
    }
    _ => Ok(unevaluated(args)),
  }
}

/// NumberDecompose[x, {u1, u2, ...}] - greedy decomposition of x into
/// multiples of the units: quotients truncate toward zero and the last
/// entry is the exact remainder over the final unit (real whenever any
/// input is a machine real). Units must be a nonincreasing list of
/// positive numbers (NumberDecompose::psv otherwise, but only once the
/// value itself is numeric); symbolic input stays unevaluated.
pub fn number_decompose_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated("NumberDecompose", args);
  if args.len() != 2 {
    return Ok(unevaluated(args));
  }

  // Exact values as normalized (numerator, positive denominator)
  enum Num {
    Exact(i128, i128),
    Real(f64),
  }
  let to_num = |e: &Expr| -> Option<Num> {
    match e {
      Expr::Integer(n) => Some(Num::Exact(*n, 1)),
      Expr::Real(r) => Some(Num::Real(*r)),
      Expr::FunctionCall { name, args } if name == "Rational" => {
        match (&args[0], &args[1]) {
          (Expr::Integer(p), Expr::Integer(q)) if *q > 0 => {
            Some(Num::Exact(*p, *q))
          }
          (Expr::Integer(p), Expr::Integer(q)) if *q < 0 => {
            Some(Num::Exact(-*p, -*q))
          }
          _ => None,
        }
      }
      _ => None,
    }
  };

  // The value must be numeric before the unit list is even validated
  let Some(value) = to_num(&args[0]) else {
    return Ok(unevaluated(args));
  };
  let unit_exprs = match &args[1] {
    Expr::List(items) if !items.is_empty() => items,
    _ => return Ok(unevaluated(args)),
  };
  let psv = |args: &[Expr]| {
    crate::emit_message(&format!(
      "NumberDecompose::psv: {} is not a list of nonincreasing positive numbers.",
      expr_to_string(&args[1])
    ));
    Ok(unevaluated(args))
  };
  let units: Vec<Num> = match unit_exprs.iter().map(&to_num).collect() {
    Some(u) => u,
    None => return psv(args),
  };
  let as_f64 = |n: &Num| -> f64 {
    match n {
      Num::Exact(p, q) => *p as f64 / *q as f64,
      Num::Real(r) => *r,
    }
  };
  let positive_nonincreasing = units.iter().all(|u| as_f64(u) > 0.0)
    && units.windows(2).all(|w| as_f64(&w[0]) >= as_f64(&w[1]));
  if !positive_nonincreasing {
    return psv(args);
  }

  let any_real = matches!(value, Num::Real(_))
    || units.iter().any(|u| matches!(u, Num::Real(_)));

  let mut result: Vec<Expr> = Vec::with_capacity(units.len());
  if any_real {
    let mut rem = as_f64(&value);
    for (i, u) in units.iter().enumerate() {
      let u = as_f64(u);
      if i + 1 == units.len() {
        result.push(Expr::Real(rem / u));
      } else {
        let q = (rem / u).trunc();
        result.push(Expr::Integer(q as i128));
        rem -= q * u;
      }
    }
  } else {
    let (mut rn, mut rd) = match value {
      Num::Exact(p, q) => (p, q),
      Num::Real(_) => unreachable!(),
    };
    for (i, u) in units.iter().enumerate() {
      let (un, ud) = match u {
        Num::Exact(p, q) => (*p, *q),
        Num::Real(_) => unreachable!(),
      };
      // rem / u = (rn*ud) / (rd*un); un > 0, rd > 0
      let qn = rn * ud;
      let qd = rd * un;
      if i + 1 == units.len() {
        result.push(make_rational_expr(qn, qd));
      } else {
        let q_int = qn / qd; // Rust integer division truncates toward zero
        result.push(Expr::Integer(q_int));
        // rem -= q_int * u
        rn = rn * ud - q_int * un * rd;
        rd *= ud;
        (rn, rd) = rat_reduce(rn, rd);
      }
    }
  }
  Ok(Expr::List(result.into()))
}

/// NumberCompose[coeffs, units] reconstructs a number from a list of
/// coefficients and a list of units (the inverse of NumberDecompose). The
/// units must be nonincreasing positive numbers, and `coeffs` cannot be longer
/// than `units`; the coefficients are paired with the trailing (last `k`)
/// units, so `NumberCompose[{1, 2, 3}, {100, 10, 1}]` is `123` and the shorter
/// `NumberCompose[{1, 2}, {100, 10, 1}]` is `12`.
pub fn number_compose_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated("NumberCompose", args);
  if args.len() != 2 {
    return Ok(unevaluated(args));
  }
  let Expr::List(coeff_exprs) = &args[0] else {
    return Ok(unevaluated(args));
  };
  let Expr::List(unit_exprs) = &args[1] else {
    return Ok(unevaluated(args));
  };

  // The coefficient list cannot be longer than the unit list. This is checked
  // before the unit list is validated (matching wolframscript).
  if coeff_exprs.len() > unit_exprs.len() {
    crate::emit_message(&format!(
      "NumberCompose::ulen: List {} of coefficients cannot be longer than list {} of units.",
      expr_to_string(&args[0]),
      expr_to_string(&args[1])
    ));
    return Ok(unevaluated(args));
  }

  // Units must be nonincreasing positive numbers.
  let to_f64 = |e: &Expr| -> Option<f64> {
    match e {
      Expr::Integer(n) => Some(*n as f64),
      Expr::Real(r) => Some(*r),
      Expr::BigInteger(b) => b.to_string().parse::<f64>().ok(),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        match (&args[0], &args[1]) {
          (Expr::Integer(p), Expr::Integer(q)) if *q != 0 => {
            Some(*p as f64 / *q as f64)
          }
          _ => None,
        }
      }
      _ => None,
    }
  };
  let psv = |args: &[Expr]| {
    crate::emit_message(&format!(
      "NumberCompose::psv: {} is not a list of nonincreasing positive numbers.",
      expr_to_string(&args[1])
    ));
    Ok(unevaluated(args))
  };
  let unit_vals: Vec<f64> = match unit_exprs.iter().map(&to_f64).collect() {
    Some(u) => u,
    None => return psv(args),
  };
  let nonincreasing_positive = unit_vals.iter().all(|&u| u > 0.0)
    && unit_vals.windows(2).all(|w| w[0] >= w[1]);
  if !nonincreasing_positive {
    return psv(args);
  }

  // Pair the coefficients with the trailing units (right-aligned) and sum the
  // products via Dot, which handles exact rational and symbolic coefficients.
  let k = coeff_exprs.len();
  let taken: Vec<Expr> = unit_exprs[unit_exprs.len() - k..].to_vec();
  crate::evaluator::evaluate_function_call_ast(
    "Dot",
    &[Expr::List(coeff_exprs.clone()), Expr::List(taken.into())],
  )
}

/// MinkowskiQuestionMark[x] - Minkowski's question-mark function via
/// the dyadic continued-fraction formula Wolfram uses:
/// ?([a0; a1, a2, ...]) = a0 + sum_k (-1)^(k+1) * 2^(1 - (a1+...+ak)),
/// fed with Wolfram's ContinuedFraction convention (termwise-negated
/// quotients for negative x, which makes exponents positive and yields
/// values like ?(-1/2) = 8), and with periodic tails of quadratic
/// irrationals summed as formal geometric series even when divergent
/// (?(-Sqrt[2]) = 3/5). Machine reals stay unevaluated: wolframscript
/// uses a limited-precision expansion there whose noise we do not
/// reproduce.
pub fn minkowski_question_mark_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated("MinkowskiQuestionMark", args);
  if args.len() != 1 {
    return Ok(unevaluated(args));
  }

  // Continued fraction (prefix, period) in Wolfram's convention
  let (prefix, period): (Vec<i128>, Vec<i128>) = match &args[0] {
    Expr::Integer(n) => return Ok(Expr::Integer(*n)),
    Expr::FunctionCall { name, args: rargs }
      if name == "Rational" && rargs.len() == 2 =>
    {
      match (&rargs[0], &rargs[1]) {
        (Expr::Integer(p), Expr::Integer(q)) if *q != 0 => {
          let negative = (*p < 0) != (*q < 0);
          let (mut a, mut b) = (p.abs(), q.abs());
          let mut terms = Vec::new();
          while b != 0 {
            let quot = a / b;
            terms.push(if negative { -quot } else { quot });
            let rem = a % b;
            a = b;
            b = rem;
          }
          (terms, Vec::new())
        }
        _ => return Ok(unevaluated(args)),
      }
    }
    other => {
      // Quadratic irrationals: GoldenRatio, Sqrt[d], (p + q*Sqrt[d])/r.
      // Compute the expansion of |x| and negate termwise when x < 0.
      let golden = matches!(other, Expr::Identifier(s) | Expr::Constant(s) if s == "GoldenRatio");
      let neg_golden = match other {
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand,
        } => matches!(
          operand.as_ref(),
          Expr::Identifier(s) | Expr::Constant(s) if s == "GoldenRatio"
        ),
        _ => false,
      };
      let quad: Option<(i128, i128, i128, i128, bool)> = if golden {
        Some((1, 1, 5, 2, false))
      } else if neg_golden {
        Some((1, 1, 5, 2, true))
      } else if let Some(d) = extract_sqrt_integer(other) {
        Some((0, 1, d, 1, false))
      } else if let Some((p, q, d, r)) = extract_quadratic_irrational(other) {
        // Sign of (p + q*sqrt(d)) / r
        let value = (p as f64 + q as f64 * (d as f64).sqrt()) / r as f64;
        if value < 0.0 {
          Some((-p, -q, d, r, true))
        } else {
          Some((p, q, d, r, false))
        }
      } else {
        None
      };
      match quad {
        Some((p, q, d, r, negative))
          if d > 0 && q != 0 && r != 0 && !is_perfect_square(d) =>
        {
          match continued_fraction_of_quadratic(p, q, d, r) {
            Some((pre, per)) if !per.is_empty() => {
              let s = if negative { -1 } else { 1 };
              (
                pre.into_iter().map(|t| s * t).collect(),
                per.into_iter().map(|t| s * t).collect(),
              )
            }
            _ => return Ok(unevaluated(args)),
          }
        }
        _ => return Ok(unevaluated(args)),
      }
    }
  };

  // Exact fraction arithmetic over BigInt
  let reduce = |num: BigInt, den: BigInt| -> (BigInt, BigInt) {
    rat_reduce_bigint(&num, &den)
  };
  let add = |a: &(BigInt, BigInt), b: &(BigInt, BigInt)| -> (BigInt, BigInt) {
    reduce(&a.0 * &b.1 + &b.0 * &a.1, &a.1 * &b.1)
  };
  // sign * 2^e as a fraction (e may be negative)
  let exp2 = |sign: i64, e: i128| -> (BigInt, BigInt) {
    if e >= 0 {
      (BigInt::from(sign) << (e as u64), BigInt::from(1))
    } else {
      (BigInt::from(sign), BigInt::from(1) << ((-e) as u64))
    }
  };

  let a0 = prefix.first().copied().unwrap_or(0);
  let mut total = (BigInt::from(a0), BigInt::from(1));
  let mut s: i128 = 0; // running sum a1 + ... + ak
  let mut sign: i64 = 1; // (-1)^(k+1)
  for a in &prefix[1..] {
    s += a;
    total = add(&total, &exp2(sign, 1 - s));
    sign = -sign;
  }

  if !period.is_empty() {
    // One cycle of terms, then the formal geometric tail
    // T / (1 - r) with r = (-1)^L * 2^(-sum(period))
    let mut t = (BigInt::from(0), BigInt::from(1));
    let mut q_sum = 0i128;
    let mut cycle_sign = sign;
    for p in &period {
      q_sum += p;
      t = add(&t, &exp2(cycle_sign, 1 - s - q_sum));
      cycle_sign = -cycle_sign;
    }
    let r_sign: i64 = if period.len() % 2 == 0 { 1 } else { -1 };
    let r = exp2(r_sign, -q_sum);
    let one_minus_r = add(&(BigInt::from(1), BigInt::from(1)), &(-r.0, r.1));
    // tail = t / (1 - r)
    let tail = reduce(&t.0 * &one_minus_r.1, &t.1 * &one_minus_r.0);
    total = add(&total, &tail);
  }

  Ok(super::number_theory::make_rational_expr(&total.0, &total.1))
}

/// FromRomanNumeral[s] - integer value of a roman numeral string.
/// Case-insensitive over I V X L C D M plus N for zero; values combine
/// with the generic pairwise subtractive rule (a symbol smaller than
/// its successor counts negative), so non-canonical forms like "IIII"
/// (4), "XIIX" (20) and "IM" (999) are accepted. Lists map elementwise.
/// Other characters emit FromRomanNumeral::nrom; non-strings emit
/// FromRomanNumeral::string.
pub fn from_roman_numeral_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated("FromRomanNumeral", args);
  if args.len() != 1 {
    return Ok(unevaluated(args));
  }
  match &args[0] {
    Expr::String(s) => {
      let values: Option<Vec<i128>> = s
        .chars()
        .map(|c| match c.to_ascii_uppercase() {
          'N' => Some(0),
          'I' => Some(1),
          'V' => Some(5),
          'X' => Some(10),
          'L' => Some(50),
          'C' => Some(100),
          'D' => Some(500),
          'M' => Some(1000),
          _ => None,
        })
        .collect();
      if let Some(v) = values {
        let mut total = 0i128;
        for (i, val) in v.iter().enumerate() {
          if v.get(i + 1).is_some_and(|next| val < next) {
            total -= val;
          } else {
            total += val;
          }
        }
        Ok(Expr::Integer(total))
      } else {
        crate::emit_message(&format!(
          "FromRomanNumeral::nrom: String {s} does not represent a valid roman numeral."
        ));
        Ok(unevaluated(args))
      }
    }
    Expr::List(items) => {
      let mut results = Vec::with_capacity(items.len());
      for item in items {
        results.push(from_roman_numeral_ast(std::slice::from_ref(item))?);
      }
      Ok(Expr::List(results.into()))
    }
    _ => {
      crate::emit_message(&format!(
        "FromRomanNumeral::string: String expected at position 1 in {}.",
        expr_to_string(&unevaluated(args))
      ));
      Ok(unevaluated(args))
    }
  }
}

/// ThueMorse[n] - the n-th Thue–Morse number: the parity (0 or 1) of the
/// number of 1-bits in the binary expansion of n. Negative integers use the
/// absolute value (ThueMorse[-5] == ThueMorse[5]). Non-integer numeric
/// arguments produce a message and stay unevaluated (matching wolframscript);
/// symbolic arguments echo silently.
pub fn thue_morse_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated("ThueMorse", args);
  if args.len() != 1 {
    return Ok(unevaluated(args));
  }
  if let Some(n) = expr_to_bigint(&args[0]) {
    let ones = n
      .abs()
      .to_radix_le(2)
      .1
      .into_iter()
      .filter(|&b| b == 1)
      .count();
    Ok(Expr::Integer((ones % 2) as i128))
  } else {
    // A Real or Rational is numeric but not a valid non-negative integer.
    if matches!(&args[0], Expr::Real(_))
      || matches!(&args[0], Expr::FunctionCall { name, .. } if name == "Rational")
    {
      crate::emit_message(&format!(
        "ThueMorse::nnintprm: Parameter {} at position 1 in {} is expected \
         to be a non-negative integer.",
        expr_to_string(&args[0]),
        expr_to_string(&unevaluated(args))
      ));
    }
    Ok(unevaluated(args))
  }
}

/// RudinShapiro[n] - the n-th Rudin–Shapiro number: (-1)^a(n), where a(n) is
/// the number of (possibly overlapping) occurrences of "11" in the binary
/// expansion of n. Defined only for non-negative integers; negative or
/// non-integer arguments stay unevaluated (matching wolframscript).
pub fn rudin_shapiro_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated("RudinShapiro", args);
  if args.len() != 1 {
    return Ok(unevaluated(args));
  }
  match expr_to_bigint(&args[0]) {
    Some(n) if !n.is_negative() => {
      // Bits from least- to most-significant; count adjacent 1,1 pairs.
      let bits = n.to_radix_le(2).1;
      let pairs = bits.windows(2).filter(|w| w[0] == 1 && w[1] == 1).count();
      Ok(Expr::Integer(if pairs % 2 == 0 { 1 } else { -1 }))
    }
    _ => {
      if matches!(&args[0], Expr::Real(_))
        || matches!(&args[0], Expr::FunctionCall { name, .. } if name == "Rational")
      {
        crate::emit_message(&format!(
          "RudinShapiro::nnintprm: Parameter {} at position 1 in {} is \
           expected to be a non-negative integer.",
          expr_to_string(&args[0]),
          expr_to_string(&unevaluated(args))
        ));
      }
      Ok(unevaluated(args))
    }
  }
}
