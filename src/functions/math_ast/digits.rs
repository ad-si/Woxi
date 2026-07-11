#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;
use num_bigint::BigInt;
use num_traits::Signed;

/// DigitCount[n] - counts of each digit 1-9,0 in base 10
/// DigitCount[n, b] - counts of each digit in base b
/// DigitCount[n, b, d] - count of specific digit d in base b
pub fn digit_count_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "DigitCount expects 1 to 3 arguments".into(),
    ));
  }
  let n = match expr_to_bigint(&args[0]) {
    Some(n) => n.abs(),
    None => {
      return Ok(Expr::FunctionCall {
        name: "DigitCount".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  let base = if args.len() >= 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => b,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "DigitCount".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    10
  };

  // Get digit list in the given base
  use num_traits::Zero;
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

  if args.len() == 3 {
    // DigitCount[n, b, d] - count of specific digit d
    let d = match expr_to_i128(&args[2]) {
      Some(d) => d as usize,
      None => {
        return Ok(Expr::FunctionCall {
          name: "DigitCount".to_string(),
          args: args.to_vec().into(),
        });
      }
    };
    let count = digits.iter().filter(|&&x| x == d).count();
    Ok(Expr::Integer(count as i128))
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
pub fn digit_sum_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "DigitSum expects 1 or 2 arguments".into(),
    ));
  }
  let n = match expr_to_bigint(&args[0]) {
    Some(n) => n.abs(),
    None => {
      return Ok(Expr::FunctionCall {
        name: "DigitSum".to_string(),
        args: args.to_vec().into(),
      });
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
    for b in bases.iter() {
      let Some(bi) = expr_to_i128(b) else {
        return Ok(Expr::FunctionCall {
          name: "DigitSum".to_string(),
          args: args.to_vec().into(),
        });
      };
      // A MixedRadix specification may legitimately contain a radix of 1
      // (e.g. MixedRadix[{60, 60, 1}]); only reject non-positive radices.
      if bi < 1 {
        return Ok(Expr::FunctionCall {
          name: "DigitSum".to_string(),
          args: args.to_vec().into(),
        });
      }
      base_vals.push(BigInt::from(bi));
    }
    use num_traits::Zero;
    let mut sum = BigInt::from(0);
    let mut val = n;
    // Right-to-left: rightmost digit uses last base, etc.
    for b in base_vals.iter().rev() {
      if val.is_zero() {
        break;
      }
      sum += &val % b;
      val /= b;
    }
    // Any remaining value is the overflow digit (no base constraint).
    sum += val;
    return Ok(bigint_to_expr(sum));
  }

  let base = if args.len() == 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => b,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "DigitSum".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    10
  };

  use num_traits::Zero;
  let big_base = BigInt::from(base);
  let mut sum = BigInt::from(0);
  let mut val = n;
  if val.is_zero() {
    return Ok(Expr::Integer(0));
  }
  while !val.is_zero() {
    sum += &val % &big_base;
    val /= &big_base;
  }
  Ok(bigint_to_expr(sum))
}

// --- Minimal arbitrary-precision unsigned integer for high-precision ContinuedFraction ---

/// Little-endian base-2^64 unsigned integer.
#[derive(Clone, Debug)]
pub struct BigUint {
  digits: Vec<u64>,
}

impl BigUint {
  fn zero() -> Self {
    Self { digits: vec![0] }
  }

  fn from_u64(n: u64) -> Self {
    Self { digits: vec![n] }
  }

  fn from_u128(n: u128) -> Self {
    let lo = n as u64;
    let hi = (n >> 64) as u64;
    let mut b = Self {
      digits: vec![lo, hi],
    };
    b.trim();
    b
  }

  fn is_zero(&self) -> bool {
    self.digits.iter().all(|&d| d == 0)
  }

  fn trim(&mut self) {
    while self.digits.len() > 1 && *self.digits.last().unwrap() == 0 {
      self.digits.pop();
    }
  }

  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let a_len = self.digits.len();
    let b_len = other.digits.len();
    if a_len != b_len {
      return a_len.cmp(&b_len);
    }
    for i in (0..a_len).rev() {
      match self.digits[i].cmp(&other.digits[i]) {
        Ordering::Equal => continue,
        ord => return ord,
      }
    }
    Ordering::Equal
  }

  /// self + other
  fn add(&self, other: &Self) -> Self {
    let max_len = self.digits.len().max(other.digits.len());
    let mut result = Vec::with_capacity(max_len + 1);
    let mut carry: u64 = 0;
    for i in 0..max_len {
      let a = if i < self.digits.len() {
        self.digits[i]
      } else {
        0
      };
      let b = if i < other.digits.len() {
        other.digits[i]
      } else {
        0
      };
      let (s1, c1) = a.overflowing_add(b);
      let (s2, c2) = s1.overflowing_add(carry);
      result.push(s2);
      carry = (c1 as u64) + (c2 as u64);
    }
    if carry > 0 {
      result.push(carry);
    }
    let mut r = Self { digits: result };
    r.trim();
    r
  }

  /// self - other (assumes self >= other)
  fn sub(&self, other: &Self) -> Self {
    let mut result = Vec::with_capacity(self.digits.len());
    let mut borrow: u64 = 0;
    for i in 0..self.digits.len() {
      let a = self.digits[i];
      let b = if i < other.digits.len() {
        other.digits[i]
      } else {
        0
      };
      let (s1, c1) = a.overflowing_sub(b);
      let (s2, c2) = s1.overflowing_sub(borrow);
      result.push(s2);
      borrow = (c1 as u64) + (c2 as u64);
    }
    let mut r = Self { digits: result };
    r.trim();
    r
  }

  /// self * other
  fn mul(&self, other: &Self) -> Self {
    let n = self.digits.len() + other.digits.len();
    let mut result = vec![0u64; n];
    for i in 0..self.digits.len() {
      let mut carry: u128 = 0;
      for j in 0..other.digits.len() {
        let prod = (self.digits[i] as u128) * (other.digits[j] as u128)
          + (result[i + j] as u128)
          + carry;
        result[i + j] = prod as u64;
        carry = prod >> 64;
      }
      if carry > 0 {
        result[i + other.digits.len()] += carry as u64;
      }
    }
    let mut r = Self { digits: result };
    r.trim();
    r
  }

  /// self * scalar
  fn mul_u64(&self, scalar: u64) -> Self {
    let mut result = Vec::with_capacity(self.digits.len() + 1);
    let mut carry: u128 = 0;
    for &d in &self.digits {
      let prod = (d as u128) * (scalar as u128) + carry;
      result.push(prod as u64);
      carry = prod >> 64;
    }
    if carry > 0 {
      result.push(carry as u64);
    }
    let mut r = Self { digits: result };
    r.trim();
    r
  }

  /// Division: returns (quotient, remainder)
  fn divmod(&self, other: &Self) -> (Self, Self) {
    use std::cmp::Ordering;
    if other.is_zero() {
      panic!("BigUint division by zero");
    }
    match self.cmp(other) {
      Ordering::Less => return (Self::zero(), self.clone()),
      Ordering::Equal => return (Self::from_u64(1), Self::zero()),
      _ => {}
    }
    if other.digits.len() == 1 {
      let d = other.digits[0];
      let (q, r) = self.divmod_u64(d);
      return (q, Self::from_u64(r));
    }
    // Long division
    self.long_divmod(other)
  }

  /// Divide by a single u64, returns (quotient, remainder)
  fn divmod_u64(&self, d: u64) -> (Self, u64) {
    let mut result = vec![0u64; self.digits.len()];
    let mut rem: u128 = 0;
    for i in (0..self.digits.len()).rev() {
      rem = (rem << 64) | (self.digits[i] as u128);
      result[i] = (rem / d as u128) as u64;
      rem %= d as u128;
    }
    let mut q = Self { digits: result };
    q.trim();
    (q, rem as u64)
  }

  /// Long division for multi-digit divisors
  fn long_divmod(&self, other: &Self) -> (Self, Self) {
    // Shift-and-subtract algorithm operating on bits
    let mut remainder = Self::zero();
    let self_bits = self.bit_len();
    let mut quotient_digits = vec![0u64; self_bits.div_ceil(64)];
    for i in (0..self_bits).rev() {
      // remainder = remainder << 1 | bit_i(self)
      remainder = remainder.shl1();
      if self.bit(i) {
        remainder.digits[0] |= 1;
      }
      if remainder.cmp(other) != std::cmp::Ordering::Less {
        remainder = remainder.sub(other);
        quotient_digits[i / 64] |= 1u64 << (i % 64);
      }
    }
    let mut q = Self {
      digits: quotient_digits,
    };
    q.trim();
    (q, remainder)
  }

  fn bit_len(&self) -> usize {
    if self.is_zero() {
      return 0;
    }
    let top = self.digits.len() - 1;
    top * 64 + (64 - self.digits[top].leading_zeros() as usize)
  }

  fn bit(&self, i: usize) -> bool {
    let word = i / 64;
    let bit = i % 64;
    if word >= self.digits.len() {
      false
    } else {
      (self.digits[word] >> bit) & 1 == 1
    }
  }

  fn shl1(&self) -> Self {
    let mut result = Vec::with_capacity(self.digits.len() + 1);
    let mut carry = 0u64;
    for &d in &self.digits {
      result.push((d << 1) | carry);
      carry = d >> 63;
    }
    if carry > 0 {
      result.push(carry);
    }
    let mut r = Self { digits: result };
    r.trim();
    r
  }

  /// Convert to i128 if it fits
  fn to_i128(&self) -> Option<i128> {
    match self.digits.len() {
      1 => Some(self.digits[0] as i128),
      2 => {
        let val = (self.digits[1] as u128) << 64 | (self.digits[0] as u128);
        if val <= i128::MAX as u128 {
          Some(val as i128)
        } else {
          None
        }
      }
      _ => None,
    }
  }

  fn gcd(a: &Self, b: &Self) -> Self {
    let mut a = a.clone();
    let mut b = b.clone();
    while !b.is_zero() {
      let (_, r) = a.divmod(&b);
      a = b;
      b = r;
    }
    a
  }
}

/// Signed big rational number: numerator/denominator with sign.
#[derive(Clone)]
pub struct BigRational {
  num: BigUint,
  den: BigUint,
  negative: bool,
}

impl BigRational {
  fn zero() -> Self {
    Self {
      num: BigUint::zero(),
      den: BigUint::from_u64(1),
      negative: false,
    }
  }

  fn from_i64(n: i64) -> Self {
    Self {
      num: BigUint::from_u64(n.unsigned_abs()),
      den: BigUint::from_u64(1),
      negative: n < 0,
    }
  }

  fn reduce(&mut self) {
    if self.num.is_zero() {
      self.negative = false;
      self.den = BigUint::from_u64(1);
      return;
    }
    let g = BigUint::gcd(&self.num, &self.den);
    if !g.is_zero() && g.digits != vec![1] {
      let (qn, _) = self.num.divmod(&g);
      let (qd, _) = self.den.divmod(&g);
      self.num = qn;
      self.den = qd;
    }
  }

  /// self + other
  fn add(&self, other: &Self) -> Self {
    // a/b + c/d = (a*d + c*b) / (b*d) respecting signs
    let ad = self.num.mul(&other.den);
    let cb = other.num.mul(&self.den);
    let bd = self.den.mul(&other.den);
    let (num, negative) = if self.negative == other.negative {
      (ad.add(&cb), self.negative)
    } else {
      match ad.cmp(&cb) {
        std::cmp::Ordering::Greater | std::cmp::Ordering::Equal => {
          (ad.sub(&cb), self.negative)
        }
        std::cmp::Ordering::Less => (cb.sub(&ad), other.negative),
      }
    };
    let mut r = Self {
      num,
      den: bd,
      negative,
    };
    r.reduce();
    r
  }

  /// self - other
  fn sub(&self, other: &Self) -> Self {
    let neg_other = Self {
      num: other.num.clone(),
      den: other.den.clone(),
      negative: !other.negative,
    };
    self.add(&neg_other)
  }

  /// self * scalar (positive integer)
  fn mul_u64(&self, s: u64) -> Self {
    let mut r = Self {
      num: self.num.mul_u64(s),
      den: self.den.clone(),
      negative: self.negative,
    };
    r.reduce();
    r
  }

  /// Floor division: returns (floor, remainder) such that self = floor + remainder/1
  /// where 0 <= remainder < 1 (for positive self)
  fn floor_and_remainder(&self) -> (i128, Self) {
    let (q, r) = self.num.divmod(&self.den);
    let q_i128 = q.to_i128().unwrap_or(0);
    let floor_val = if self.negative && !r.is_zero() {
      -(q_i128 + 1)
    } else if self.negative {
      -q_i128
    } else {
      q_i128
    };
    // remainder = self - floor_val
    let floor_rat = Self::from_i64(floor_val as i64);
    let rem = self.sub(&floor_rat);
    (floor_val, rem)
  }

  /// 1 / self
  fn reciprocal(&self) -> Self {
    Self {
      num: self.den.clone(),
      den: self.num.clone(),
      negative: self.negative,
    }
  }
}

/// Compute atan(1/k) as a BigRational using the Taylor series:
/// atan(x) = x - x^3/3 + x^5/5 - ...
/// For x = 1/k: atan(1/k) = 1/k - 1/(3*k^3) + 1/(5*k^5) - ...
fn big_atan_recip(k: u64, terms: usize) -> BigRational {
  let mut result = BigRational::zero();
  let k2 = k as u128 * k as u128; // k^2 as u128
  // power_denom tracks k^(2n+1) as BigUint
  let mut power_denom = BigUint::from_u64(k);
  let k2_big = BigUint::from_u128(k2);
  for n in 0..terms {
    let divisor = (2 * n + 1) as u64;
    // term = 1 / (divisor * power_denom)
    let term = BigRational {
      num: BigUint::from_u64(1),
      den: power_denom.mul_u64(divisor),
      negative: n % 2 != 0,
    };
    result = result.add(&term);
    power_denom = power_denom.mul(&k2_big);
  }
  result
}

/// Compute Pi as a BigRational using Machin's formula:
/// Pi/4 = 4*atan(1/5) - atan(1/239)
fn pi_as_big_rational(terms: usize) -> BigRational {
  let atan5 = big_atan_recip(5, terms);
  let atan239 = big_atan_recip(239, terms);
  // Pi = 4 * (4*atan(1/5) - atan(1/239))
  let four_atan5 = atan5.mul_u64(4);
  let diff = four_atan5.sub(&atan239);
  diff.mul_u64(4)
}

/// Compute E as a BigRational using the series: e = sum(1/k!, k=0..terms)
fn e_as_big_rational(terms: usize) -> BigRational {
  let mut result = BigRational::zero();
  let mut factorial = BigUint::from_u64(1);
  for k in 0..terms {
    if k > 0 {
      factorial = factorial.mul_u64(k as u64);
    }
    let term = BigRational {
      num: BigUint::from_u64(1),
      den: factorial.clone(),
      negative: false,
    };
    result = result.add(&term);
  }
  result
}

/// Compute the continued fraction of a BigRational, returning up to n terms.
fn continued_fraction_from_big_rational(
  val: &BigRational,
  n: usize,
) -> Vec<i128> {
  let mut result = Vec::new();
  let mut current = val.clone();
  for _ in 0..n {
    let (floor_val, rem) = current.floor_and_remainder();
    result.push(floor_val);
    if rem.num.is_zero() {
      break;
    }
    current = rem.reciprocal();
  }
  result
}

/// Try to compute a constant expression as a high-precision BigRational.
/// Returns None if the expression is not a recognized constant.
fn try_constant_as_big_rational(
  expr: &Expr,
  n_terms: usize,
) -> Option<BigRational> {
  // Use n_terms + 10 series terms for safety margin
  let series_terms = n_terms + 10;
  match expr {
    Expr::Constant(name) if name == "Pi" => {
      Some(pi_as_big_rational(series_terms))
    }
    Expr::Constant(name) if name == "E" => {
      Some(e_as_big_rational(series_terms))
    }
    Expr::Constant(name) if name == "-Pi" => {
      let mut pi = pi_as_big_rational(series_terms);
      pi.negative = true;
      Some(pi)
    }
    Expr::Constant(name) if name == "-E" => {
      let mut e = e_as_big_rational(series_terms);
      e.negative = true;
      Some(e)
    }
    _ => None,
  }
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
      op: crate::syntax::BinaryOperator::Power,
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
      op: crate::syntax::BinaryOperator::Power,
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
      op: crate::syntax::BinaryOperator::Divide,
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
        _ => Expr::FunctionCall {
          name: "Times".to_string(),
          args: remaining.into(),
        },
      };
      (inner, scale_n, scale_d)
    }
    _ => (expr.clone(), 1, 1),
  };
  let inner = &inner_owned;
  // Inner: expect Plus[int, (q·Sqrt[d])] or just Sqrt[d] or k·Sqrt[d]
  let mut p: i128 = 0;
  let mut q: i128 = 0;
  let mut d: i128 = 0;
  let terms: Vec<&Expr> = match inner {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left,
      right,
    } => vec![left.as_ref(), right.as_ref()],
    Expr::FunctionCall { name, args } if name == "Plus" => {
      args.iter().collect()
    }
    _ => vec![inner],
  };
  for t in &terms {
    if let Expr::Integer(n) = t {
      p = p.checked_add(*n)?;
      continue;
    }
    if let Some(dd) = extract_sqrt_integer(t) {
      q = q.checked_add(1)?;
      if d != 0 && d != dd {
        return None;
      }
      d = dd;
      continue;
    }
    // k * Sqrt[d] — canonically as Times[k, Sqrt[d]].
    let (coeff, sqrt_expr) = match t {
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left,
        right,
      } => (left.as_ref(), right.as_ref()),
      Expr::FunctionCall { name, args }
        if name == "Times" && args.len() == 2 =>
      {
        (&args[0], &args[1])
      }
      _ => return None,
    };
    let k = match coeff {
      Expr::Integer(n) => *n,
      _ => return None,
    };
    let dd = extract_sqrt_integer(sqrt_expr)?;
    q = q.checked_add(k)?;
    if d != 0 && d != dd {
      return None;
    }
    d = dd;
  }
  if q == 0 || d == 0 {
    return None;
  }
  // Compose with the outer scale: (scale_num/scale_den) · (p + q√d)
  //   = (scale_num·p + scale_num·q·√d) / scale_den.
  let final_p = p.checked_mul(scale_num)?;
  let final_q = q.checked_mul(scale_num)?;
  let final_r = scale_den;
  Some((final_p, final_q, d, final_r))
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
  // Normalize to the form `(P + √N) / Q` with `Q | (N − P²)`.
  // Starting from `(p + q √d) / r`: set P = p · sign(r), Q = |r|,
  // N = q² · d, then if `Q ∤ (N − P²)` multiply P and Q by |Q| (and N by Q²)
  // until divisibility holds.
  let sign = if r < 0 { -1i128 } else { 1 };
  let mut big_p: i128 = p.checked_mul(sign)?;
  // Merge the sign of q by absorbing it into the root: we rely on q·√d
  // appearing with the sign carried into q. When q < 0, we need to adjust:
  // (p + q √d) / r with q < 0 is the same as (p - |q| √d)/r — but CF of a
  // negative-Sqrt combination is harder to handle uniformly. Bail if q < 0.
  if q <= 0 {
    return None;
  }
  let mut big_n: i128 = q.checked_mul(q)?.checked_mul(d)?;
  let mut big_q: i128 = r.checked_mul(sign)?;
  if big_q <= 0 {
    return None;
  }
  // Ensure Q divides (N − P²); multiply through by |Q| if not.
  for _ in 0..4 {
    let diff = big_n.checked_sub(big_p.checked_mul(big_p)?)?;
    if diff.rem_euclid(big_q) == 0 {
      break;
    }
    // Scale P ← P · Q, N ← N · Q², Q ← Q · |Q|.
    let new_p = big_p.checked_mul(big_q)?;
    let q_sq = big_q.checked_mul(big_q)?;
    let new_n = big_n.checked_mul(q_sq)?;
    let new_q = big_q.checked_mul(big_q)?;
    big_p = new_p;
    big_n = new_n;
    big_q = new_q;
  }

  // Iterate: a_k = floor((P + √N) / Q); M = P − a·Q;
  //          P' = −M; Q' = (N − M²) / Q.
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
    // a = floor((P + sqrt(N)) / Q). Since Q > 0, use integer floor:
    // a = floor((P + sqrt_N_floor) / Q) when P + sqrt_N_floor > 0,
    // but we also need to handle the case where sqrt(N) is not integer and
    // the approximation rounds down. Using `sqrt_n_floor` works as a
    // lower bound for sqrt(N); division floor then yields the correct a.
    let num = p_k + sqrt_n_floor;
    let a = num.div_euclid(q_k);
    terms.push(a);
    let m = p_k - a * q_k;
    let new_q = (big_n - m * m) / q_k;
    if new_q == 0 {
      return None;
    }
    p_k = -m;
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
    let unevaluated = || {
      Ok(Expr::FunctionCall {
        name: "ContinuedFraction".to_string(),
        args: args.to_vec().into(),
      })
    };
    let n_cap: Option<usize> = if args.len() == 2 {
      match &args[1] {
        Expr::Integer(n) if *n > 0 => Some(*n as usize),
        _ => return unevaluated(),
      }
    } else {
      None
    };
    let rat = crate::evaluator::evaluate_function_call_ast(
      "Rationalize",
      std::slice::from_ref(&args[0]),
    )?;
    if matches!(rat, Expr::Integer(_)) {
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
        return Ok(Expr::FunctionCall {
          name: "ContinuedFraction".to_string(),
          args: args.to_vec().into(),
        });
      }
    };

    // Try high-precision computation for known constants
    if let Some(big_rat) = try_constant_as_big_rational(&args[0], n) {
      let cf = continued_fraction_from_big_rational(&big_rat, n);
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

  Ok(Expr::FunctionCall {
    name: "ContinuedFraction".to_string(),
    args: args.to_vec().into(),
  })
}

/// Greatest common divisor of two non-negative integers.
fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
  a = a.abs();
  b = b.abs();
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
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
  fn new(num: i128, den: i128) -> Rat {
    let mut r = Rat { num, den };
    r.reduce();
    r
  }
  fn int(n: i128) -> Rat {
    Rat { num: n, den: 1 }
  }
  fn reduce(&mut self) {
    if self.den < 0 {
      self.num = -self.num;
      self.den = -self.den;
    }
    let g = gcd_i128(self.num, self.den);
    if g > 1 {
      self.num /= g;
      self.den /= g;
    }
    if self.num == 0 {
      self.den = 1;
    }
  }
  fn add(self, o: Rat) -> Rat {
    Rat::new(self.num * o.den + o.num * self.den, self.den * o.den)
  }
  fn mul(self, o: Rat) -> Rat {
    Rat::new(self.num * o.num, self.den * o.den)
  }
  fn div(self, o: Rat) -> Rat {
    Rat::new(self.num * o.den, self.den * o.num)
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
    let g = gcd_i128(pn, q);
    let (pn, q) = (pn / g, q / g);
    return Some(if q == 1 {
      Expr::Integer(pn)
    } else {
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(pn), Expr::Integer(q)].into(),
      }
    });
  }
  // rn*sqrt(disc) = sign(rn) * sqrt(rn^2 * disc) = sign(rn) * f * sqrt(d).
  let inner = rn.checked_mul(rn)?.checked_mul(disc)?;
  let (f, d) = pull_square_factor(inner);
  let s = rn.signum() * f;
  if d == 1 {
    // Perfect square under the root: the value is rational.
    let total = pn + s;
    let g = gcd_i128(total, q);
    let (total, q) = (total / g, q / g);
    return Some(if q == 1 {
      Expr::Integer(total)
    } else {
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(total), Expr::Integer(q)].into(),
      }
    });
  }
  // Reduce (P, S, Q) by their common factor so D stays square-free and the
  // fraction is in lowest terms, matching wolframscript.
  let g = gcd_i128(gcd_i128(pn, s), q);
  let (pn, s, q) = (pn / g, s / g, q / g);

  // Build (pn + s*Sqrt[d])/q and let the evaluator canonicalize the surd.
  let sqrt_d = Expr::FunctionCall {
    name: "Sqrt".to_string(),
    args: vec![Expr::Integer(d)].into(),
  };
  let s_sqrt = if s == 1 {
    sqrt_d
  } else {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(s), sqrt_d].into(),
    }
  };
  let numer = if pn == 0 {
    s_sqrt
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![Expr::Integer(pn), s_sqrt].into(),
    }
  };
  let value = if q == 1 {
    numer
  } else {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        numer,
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![Expr::Integer(q), Expr::Integer(-1)].into(),
        },
      ]
      .into(),
    }
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

  let elements = match &args[0] {
    Expr::List(elems) => elems,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "FromContinuedFraction".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  if elements.is_empty() {
    return Ok(Expr::Identifier("Infinity".to_string()));
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
    return Ok(Expr::FunctionCall {
      name: "FromContinuedFraction".to_string(),
      args: args.to_vec().into(),
    });
  }

  // Collect all integers
  let mut ints: Vec<i128> = Vec::new();
  for elem in elements {
    match elem {
      Expr::Integer(n) => ints.push(*n),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "FromContinuedFraction".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  }

  // Build fraction from right to left: start with last element, then a_i + 1/acc
  // Use numerator/denominator representation to stay exact
  let mut num = *ints.last().unwrap();
  let mut den: i128 = 1;

  for i in (0..ints.len() - 1).rev() {
    // acc = num/den, we want ints[i] + 1/acc = ints[i] + den/num = (ints[i]*num + den)/num
    let new_num = ints[i] * num + den;
    let new_den = num;
    num = new_num;
    den = new_den;
  }

  // Simplify by GCD
  fn gcd(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
      let t = b;
      b = a % b;
      a = t;
    }
    a
  }

  let g = gcd(num, den);
  num /= g;
  den /= g;

  // Normalize sign: keep denominator positive
  if den < 0 {
    num = -num;
    den = -den;
  }

  if den == 1 {
    Ok(Expr::Integer(num))
  } else {
    Ok(Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(num), Expr::Integer(den)].into(),
    })
  }
}

/// IntegerDigits[n], IntegerDigits[n, b], IntegerDigits[n, b, len]
pub fn integer_digits_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "IntegerDigits expects 1 to 3 arguments".into(),
    ));
  }

  let unevaluated = || Expr::FunctionCall {
    name: "IntegerDigits".to_string(),
    args: args.to_vec().into(),
  };
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

  use num_traits::Zero;

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
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<(Vec<i128>, i64), InterpreterError> {
  use astro_float::BigFloat;
  if bf.is_zero() {
    return Ok((vec![0; num_digits.max(1)], 0));
  }
  if base < 2 {
    return Err(InterpreterError::EvaluationError(
      "RealDigits: base must be >= 2".into(),
    ));
  }
  let base_bf = BigFloat::from_i128(base, bits);
  let one = BigFloat::from_i128(1, bits);
  let mut x = bf.clone();
  // Normalize into [1, base) by dividing/multiplying by base.
  let mut shift: i64 = 0;
  while x.cmp(&base_bf).map(|v| v >= 0).unwrap_or(false) {
    x = x.div(&base_bf, bits, rm);
    shift += 1;
  }
  while x.cmp(&one).map(|v| v < 0).unwrap_or(false) {
    x = x.mul(&base_bf, bits, rm);
    shift -= 1;
  }
  // Wolfram's exponent = shift + 1: after normalization x ∈ [1, base),
  // so the most-significant digit sits at position `shift`, and there are
  // `shift + 1` digits strictly to the left of the radix point.
  let exponent = shift + 1;

  // Extract `num_digits` base-`base` digits by repeated floor-and-shift.
  let mut digits: Vec<i128> = Vec::with_capacity(num_digits);
  for _ in 0..num_digits {
    let int_part = x.floor();
    let digit = bigfloat_small_int_to_i128(&int_part, rm, cc)?;
    digits.push(digit);
    // x = (x - digit) * base
    let digit_bf = BigFloat::from_i128(digit, bits);
    x = x.sub(&digit_bf, bits, rm);
    x = x.mul(&base_bf, bits, rm);
  }
  Ok((digits, exponent))
}

/// Convert a small non-negative BigFloat (representing an integer that fits in
/// i128) to i128 via decimal formatting. Used for extracting single digits in
/// arbitrary-base `RealDigits`.
fn bigfloat_small_int_to_i128(
  bf: &astro_float::BigFloat,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<i128, InterpreterError> {
  if bf.is_zero() {
    return Ok(0);
  }
  let s = bf.format(astro_float::Radix::Dec, rm, cc).map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "RealDigits: format error: {:?}",
      e
    ))
  })?;
  // astro-float `format` returns mantissa-style strings such as "3.14e1" or
  // "0.0". For a value that is already an exact integer we get a string like
  // "3.0e0" or "3e0". Strip any fractional part (it should be zero because
  // the caller passed `floor(x)`) and parse the implicit integer.
  let lower = s.to_ascii_lowercase();
  let (mantissa_str, exp_part): (&str, i64) = if let Some(idx) = lower.find('e')
  {
    let (m, e) = s.split_at(idx);
    let e_trim = &e[1..];
    let exp: i64 = e_trim.parse().map_err(|_| {
      InterpreterError::EvaluationError(format!(
        "RealDigits: failed to parse exponent in {}",
        s
      ))
    })?;
    (m, exp)
  } else {
    (s.as_str(), 0)
  };
  let mantissa_str = mantissa_str.trim_start_matches('+');
  let (sign, body) = if let Some(rest) = mantissa_str.strip_prefix('-') {
    (-1i128, rest)
  } else {
    (1i128, mantissa_str)
  };
  let (int_str, frac_str) = match body.find('.') {
    Some(i) => (&body[..i], &body[i + 1..]),
    None => (body, ""),
  };
  // Combine integer and fractional parts without the decimal point, tracking
  // how the `.` shifted the exponent.
  let digits_str: String = int_str.chars().chain(frac_str.chars()).collect();
  let frac_len = frac_str.len() as i64;
  let effective_exp = exp_part - frac_len;
  let base_int: i128 = digits_str.parse().map_err(|_| {
    InterpreterError::EvaluationError(format!(
      "RealDigits: failed to parse digits in {}",
      s
    ))
  })?;
  let mut value = base_int;
  if effective_exp > 0 {
    for _ in 0..effective_exp {
      value = value.checked_mul(10).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "RealDigits: digit value overflow".into(),
        )
      })?;
    }
  } else if effective_exp < 0 {
    for _ in 0..(-effective_exp) {
      // floor(x) was passed in, so this should divide cleanly.
      value /= 10;
    }
  }
  Ok(sign * value)
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
        "RealDigits: conversion error: {}",
        e
      ))
    })?;

  if digits.is_empty() || digits.iter().all(|&d| d == 0) {
    return Ok((vec![b'0'], 0));
  }

  // Convert digit values (0-9) to ASCII bytes
  let ascii_digits: Vec<u8> = digits.iter().map(|&d| b'0' + d).collect();

  Ok((ascii_digits, exponent as i64))
}

/// Extract numerator and denominator from a Rational expression.
/// Returns Some((numer, denom)) for Rational[n,d] or Integer n, None otherwise.
fn extract_rational_for_digits(expr: &Expr) -> Option<(i128, i128)> {
  match expr {
    Expr::Integer(n) => Some((*n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(a), Expr::Integer(b)) = (&args[0], &args[1]) {
        Some((*a, *b))
      } else {
        None
      }
    }
    _ => None,
  }
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
  let s = format!("{:?}", r);
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
    let combined = format!("{}{}", int_part, frac_part);
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

  let exponent: i128;
  if integer_part == 0 {
    // For numbers < 1, exponent is 0 or negative
    // Count leading zeros after decimal point
    let mut temp = n * base;
    let mut leading_zeros: i128 = 0;
    while temp < d && temp > 0 {
      leading_zeros += 1;
      temp *= base;
    }
    exponent = -leading_zeros;
  } else {
    // Count digits in integer part
    let mut count = 0i128;
    let mut temp = integer_part;
    while temp > 0 {
      count += 1;
      temp /= base;
    }
    exponent = count;
  }

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
pub fn real_digits_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 4 {
    return Err(InterpreterError::EvaluationError(
      "RealDigits expects 1 to 4 arguments".into(),
    ));
  }
  // Optional 4th argument: the position of the most-significant returned digit
  // (as the exponent of `base`, so Pi at position −3 means the 10^−3 place).
  let start_pos: Option<i128> = if args.len() == 4 {
    match expr_to_i128(&args[3]) {
      Some(n) => Some(n),
      None => {
        return Err(InterpreterError::EvaluationError(
          "RealDigits: starting position must be an integer".into(),
        ));
      }
    }
  } else {
    None
  };

  let base: i128 = if args.len() >= 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => b,
      Some(_) => {
        return Err(InterpreterError::EvaluationError(
          "RealDigits: base must be an integer >= 2".into(),
        ));
      }
      None => {
        return Err(InterpreterError::EvaluationError(
          "RealDigits: base must be an integer".into(),
        ));
      }
    }
  } else {
    10
  };

  let explicit_num_digits = args.len() >= 3;

  let num_digits: usize = if explicit_num_digits {
    match expr_to_i128(&args[2]) {
      Some(n) if n > 0 => n as usize,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "RealDigits: number of digits must be a positive integer".into(),
        ));
      }
    }
  } else if matches!(&args[0], Expr::Real(_)) && base != 10 {
    // A machine-precision Real carries ~53 bits of information; express that
    // as a digit count in the target base: ceil(53 / log2(base)). For base 2
    // that's 53 (matching wolframscript); for base 10 the decimal path's 16
    // is used instead.
    let ratio = 53.0_f64 / (base as f64).log2();
    (ratio.ceil() as usize).max(1)
  } else {
    // Default: use machine-precision (~16 digits)
    16
  };

  let expr = &args[0];

  // Determine sign and work with absolute value
  let is_negative = match expr {
    Expr::Integer(n) => *n < 0,
    Expr::Real(f) => *f < 0.0,
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      ..
    } => true,
    _ => false,
  };

  let abs_expr = if is_negative {
    Expr::FunctionCall {
      name: "Abs".to_string(),
      args: vec![expr.clone()].into(),
    }
  } else {
    expr.clone()
  };

  // Check for exact zero
  let is_zero = matches!(&abs_expr, Expr::Integer(0))
    || matches!(&abs_expr, Expr::Real(f) if *f == 0.0);

  if is_zero {
    let count = if explicit_num_digits { num_digits } else { 1 };
    let exp = if explicit_num_digits { 0 } else { 1 };
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
    rational_from_real.or_else(|| extract_rational_for_digits(&abs_expr))
    && denom != 0
  {
    if !treat_as_explicit {
      let (digit_list, exponent) =
        real_digits_rational_base(numer, denom, base);
      return Ok(Expr::List(
        vec![Expr::List(digit_list.into()), Expr::Integer(exponent)].into(),
      ));
    } else {
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
        digit_exprs.push(Expr::Identifier("Indeterminate".to_string()));
      }
      return Ok(Expr::List(
        vec![Expr::List(digit_exprs.into()), Expr::Integer(exponent)].into(),
      ));
    }
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
    InterpreterError::EvaluationError(format!("BigFloat init error: {}", e))
  })?;
  let rm = RoundingMode::ToEven;
  let bits = nominal_bits(precision);

  let bf = match expr_to_bigfloat(&abs_expr, bits, rm, &mut cc) {
    Ok(v) => v,
    Err(_) => {
      // Non-numeric expression (e.g. bare symbol): return unevaluated.
      return Ok(Expr::FunctionCall {
        name: "RealDigits".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Non-decimal bases route through the arbitrary-base digit extractor.
  if base != 10 {
    // We need enough raw digits to cover both leading zeros skipped by
    // `start_pos` (when < MSD) and the requested count. The normalization
    // loop also benefits from a slightly larger precision window.
    let extract_count = num_digits + leading_skip.unwrap_or(0) + 4;
    let (raw, base_exp) =
      bigfloat_to_digits_base(&bf, base, extract_count, bits, rm, &mut cc)?;
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
    while digits.len() < num_digits {
      digits.push(0);
    }
    digits.truncate(num_digits);
    let digit_exprs: Vec<Expr> =
      digits.iter().map(|&d| Expr::Integer(d)).collect();
    return Ok(Expr::List(
      vec![
        Expr::List(digit_exprs.into()),
        Expr::Integer(base_exp as i128),
      ]
      .into(),
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

  // Pad with zeros if we don't have enough digits
  while digits.len() < num_digits {
    digits.push(0);
  }

  // Truncate to requested number of digits
  digits.truncate(num_digits);

  let digit_exprs: Vec<Expr> =
    digits.iter().map(|&d| Expr::Integer(d)).collect();

  Ok(Expr::List(
    vec![
      Expr::List(digit_exprs.into()),
      Expr::Integer(decimal_exp as i128),
    ]
    .into(),
  ))
}

/// FromDigits[list
pub fn from_digits_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "FromDigits expects 1 or 2 arguments".into(),
    ));
  }

  // The base may be an integer with |b| >= 2 (fast numeric path; negative
  // bases compute the "negabinary"/"negadecimal" representation) or a
  // symbolic expression (e.g. `FromDigits[{1,2,3}, x]` => `3 + 2*x + x^2`).
  let numeric_base: Option<i128> = if args.len() == 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b.abs() >= 2 => Some(b),
      Some(_) => {
        return Err(InterpreterError::EvaluationError(
          "FromDigits: base must be an integer with |base| >= 2".into(),
        ));
      }
      None => None,
    }
  } else {
    Some(10)
  };

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
    let digits = match &args[0] {
      Expr::List(items) => items,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "FromDigits".to_string(),
          args: args.to_vec().into(),
        });
      }
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
      weights[n - 1 - j] = cur.clone();
    }
    let mut terms: Vec<Expr> = Vec::with_capacity(n);
    for (i, digit) in digits.iter().enumerate() {
      if weights[i] == BigInt::from(0) {
        continue;
      }
      if weights[i] == BigInt::from(1) {
        terms.push(digit.clone());
      } else {
        terms.push(Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![digit.clone(), bigint_to_expr(weights[i].clone())].into(),
        });
      }
    }
    let sum = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    };
    return crate::evaluator::evaluate_expr_to_expr(&sum);
  }

  // Symbolic base path: build an expanded polynomial sum in the base.
  // For `FromDigits[{d0, d1, ..., d_{n-1}}, b]` we produce
  //   d0*b^(n-1) + d1*b^(n-2) + ... + d_{n-2}*b + d_{n-1}.
  if numeric_base.is_none() {
    let items = match &args[0] {
      Expr::List(items) => items.clone(),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "FromDigits".to_string(),
          args: args.to_vec().into(),
        });
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
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![base_expr.clone(), Expr::Integer(power)].into(),
          }
        };
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![item.clone(), base_pow].into(),
        }
      };
      terms.push(term);
    }
    let sum = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    };
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
        return Ok(Expr::FunctionCall {
          name: "FromDigits".to_string(),
          args: args.to_vec().into(),
        });
      };
      // Wolfram allows digit values >= base (overflow digits)
      // e.g. FromDigits["a0"] in base 10 gives 10*10+0 = 100
      result = result * &big_base + BigInt::from(d);
    }
    return Ok(bigint_to_expr(result));
  }

  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "FromDigits".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

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
    } else {
      // Need a rational: int_val / base^(-shift)
      let denom = big_base.pow((-shift) as u32);
      // Simplify the fraction using Euclidean gcd
      fn gcd_bigint(a: &BigInt, b: &BigInt) -> BigInt {
        let mut a = if a < &BigInt::from(0) { -a } else { a.clone() };
        let mut b = if b < &BigInt::from(0) { -b } else { b.clone() };
        while b > BigInt::from(0) {
          let t = &a % &b;
          a = b;
          b = t;
        }
        a
      }
      let g = gcd_bigint(&int_val, &denom);
      let num = &int_val / &g;
      let den = &denom / &g;
      if den == BigInt::from(1) {
        return Ok(bigint_to_expr(num));
      } else {
        return Ok(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![bigint_to_expr(num), bigint_to_expr(den)].into(),
        });
      }
    }
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
      result = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![base_expr.clone(), result].into(),
          },
          item.clone(),
        ]
        .into(),
      };
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
    return Ok(Expr::FunctionCall {
      name: "IntegerLength".to_string(),
      args: args.to_vec().into(),
    });
  }

  let base_i128 = if args.len() == 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => b,
      Some(b) => {
        crate::emit_message(&format!(
          "IntegerLength::ibase: Base {} is not an integer greater than 1.",
          b
        ));
        return Ok(Expr::FunctionCall {
          name: "IntegerLength".to_string(),
          args: args.to_vec().into(),
        });
      }
      None => {
        return Ok(Expr::FunctionCall {
          name: "IntegerLength".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    10
  };

  // Try BigInt path first (handles both Integer and BigInteger)
  if let Some(n) = expr_to_bigint(&args[0]) {
    use num_traits::Zero;
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

  Ok(Expr::FunctionCall {
    name: "IntegerLength".to_string(),
    args: args.to_vec().into(),
  })
}

/// IntegerReverse[n] - reverse the digits of an integer in base 10.
/// IntegerReverse[n, b] - reverse the digits of n in base b.
pub fn integer_reverse_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "IntegerReverse expects 1 or 2 arguments".into(),
    ));
  }

  let base = if args.len() == 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => b,
      Some(_) => {
        return Err(InterpreterError::EvaluationError(
          "IntegerReverse: base must be at least 2".into(),
        ));
      }
      None => {
        return Ok(Expr::FunctionCall {
          name: "IntegerReverse".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    10
  };

  // Handle BigInteger
  if let Some(n) = expr_to_bigint(&args[0]) {
    use num_traits::Zero;
    let mut abs_n = if n < BigInt::zero() { -n } else { n };
    let base_big = BigInt::from(base);
    let mut result = BigInt::zero();
    while abs_n > BigInt::zero() {
      result = result * &base_big + (&abs_n % &base_big);
      abs_n /= &base_big;
    }
    return Ok(bigint_to_expr(result));
  }

  Ok(Expr::FunctionCall {
    name: "IntegerReverse".to_string(),
    args: args.to_vec().into(),
  })
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
      format!("negative {}", word)
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
      parts.push(format!("{}", group));
    } else {
      // Higher groups: use digits + scale word
      let scale = if sidx < SCALES.len() {
        SCALES[sidx]
      } else {
        return None;
      };
      parts.push(format!("{} {}", group, scale));
    }
  }

  let result = parts.join(" ");
  Some(if negative {
    format!("negative {}", result)
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
      parts.push(format!("{} {}", words, scale));
    }
  }

  let result = parts.join(", ");
  Some(if negative {
    format!("negative {}", result)
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
    | "achtzig" | "neunzig" | "hundert" | "tausend" => format!("{}ste", word),
    // Teens (dreizehn..neunzehn).
    _ => format!("{}te", word),
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
      .map(|i| i + out[i..].chars().next().unwrap().len_utf8())
      .unwrap_or(0);
    let last = out[split..].to_string();
    let suffixed = if is_german_scale_word(&last) {
      format!("{}ste", last)
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
    format!("minus {}", body)
  } else {
    body
  })
}

pub fn integer_name_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "IntegerName".to_string(),
      args: args.to_vec().into(),
    })
  };

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

  let n = match expr_to_i128(&args[0]) {
    Some(v) => v,
    None => return unevaluated(),
  };

  let cardinal = match integer_name_cardinal(n) {
    Some(s) => s,
    None => return unevaluated(),
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
        for it in items.iter() {
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
      return match integer_name_german(n, want_ordinal) {
        Some(s) => Ok(Expr::String(s)),
        None => {
          // wolframscript spells German only below 10^18; larger values warn.
          if n.unsigned_abs() >= 1_000_000_000_000_000_000 {
            crate::emit_message(&format!(
              "IntegerName::outrange: Number {} is too large for the \
               algorithms available for German.",
              n
            ));
          }
          unevaluated()
        }
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

  let n = match expr_to_i128(&args[0]) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: "RomanNumeral".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  if n == 0 {
    return Ok(Expr::String("N".to_string()));
  }

  let abs_n = n.unsigned_abs() as u64;

  // For values >= 5000, Wolfram uses display forms with overscript bars.
  // We support up to 4999 with plain Roman numerals.
  if abs_n >= 5000 {
    return Ok(Expr::FunctionCall {
      name: "RomanNumeral".to_string(),
      args: args.to_vec().into(),
    });
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

  let elements = match &cf_list {
    Expr::List(elems) => elems,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Convergents".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  if elements.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  // Collect all integers
  let mut ints: Vec<i128> = Vec::new();
  for elem in elements {
    match elem {
      Expr::Integer(n) => ints.push(*n),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Convergents".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  }

  // Use the recurrence relation:
  // h[-1] = 1, h[0] = a0
  // k[-1] = 0, k[0] = 1
  // h[n] = a[n] * h[n-1] + h[n-2]
  // k[n] = a[n] * k[n-1] + k[n-2]
  let mut result = Vec::new();
  let mut h_prev2: i128 = 1;
  let mut h_prev1: i128 = ints[0];
  let mut k_prev2: i128 = 0;
  let mut k_prev1: i128 = 1;

  result.push(make_rational_expr(h_prev1, k_prev1));

  for i in 1..ints.len() {
    let a = ints[i];
    let h = a * h_prev1 + h_prev2;
    let k = a * k_prev1 + k_prev2;
    result.push(make_rational_expr(h, k));
    h_prev2 = h_prev1;
    h_prev1 = h;
    k_prev2 = k_prev1;
    k_prev1 = k;
  }

  Ok(Expr::List(result.into()))
}

fn make_rational_expr(num: i128, den: i128) -> Expr {
  if den == 1 {
    Expr::Integer(num)
  } else if den == -1 {
    Expr::Integer(-num)
  } else {
    let g = gcd_convergents(num.abs(), den.abs());
    let (n, d) = (num / g, den / g);
    if d < 0 {
      if -d == 1 {
        Expr::Integer(-n)
      } else {
        Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(Expr::Integer(-n)),
          right: Box::new(Expr::Integer(-d)),
        }
      }
    } else if d == 1 {
      Expr::Integer(n)
    } else {
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Integer(n)),
        right: Box::new(Expr::Integer(d)),
      }
    }
  }
}

fn gcd_convergents(a: i128, b: i128) -> i128 {
  if b == 0 { a } else { gcd_convergents(b, a % b) }
}

/// NumberDigit[x, n] — returns the digit at position n of a real number x.
/// Position 0 is the ones digit, positive n goes left (tens, hundreds, ...),
/// negative n goes right (tenths, hundredths, ...).
pub fn number_digit_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use num_bigint::BigUint;
  use num_traits::{One, ToPrimitive};
  let unevaluated = || Expr::FunctionCall {
    name: "NumberDigit".to_string(),
    args: args.to_vec().into(),
  };
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
        let s = format!("{:e}", a); // like 2.10345e2
        let (mant, exp10) = s.split_once('e').unwrap();
        let exp10: i64 = exp10.parse().unwrap();
        let digits: Vec<u8> = mant
          .bytes()
          .filter(|b| b.is_ascii_digit())
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
      match crate::functions::math_ast::numerical::expr_to_bigfloat(
        other, bits, rm, &mut cc,
      ) {
        Ok(bf) => {
          let Some(radix) = radix else {
            return Ok(unevaluated());
          };
          if bf.is_zero() {
            Value::Zero
          } else {
            match bf.convert_to_radix(radix, rm, &mut cc) {
              Ok((_, digits, exponent)) => {
                Value::Radix(digits, exponent as i64)
              }
              Err(_) => return Ok(unevaluated()),
            }
          }
        }
        Err(_) => {
          crate::emit_message(&format!(
            "NumberDigit::num: Argument {} should be a number.",
            crate::syntax::format_expr(
              &args[0],
              crate::syntax::ExprForm::Output
            )
          ));
          return Ok(unevaluated());
        }
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
          Expr::Identifier("Indeterminate".to_string())
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
          return Expr::Identifier("Indeterminate".to_string());
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
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "NumberExpand".to_string(),
    args: args.to_vec().into(),
  };
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated(args));
  }
  // Base validation happens before the value is looked at
  let base: i128 = if args.len() == 2 {
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

  match &args[0] {
    Expr::Integer(n) => {
      if *n == 0 {
        return Ok(Expr::List(vec![Expr::Integer(0)].into()));
      }
      let sign = n.signum();
      let mut digits: Vec<i128> = Vec::new();
      let mut num = n.unsigned_abs();
      let base_u = base as u128;
      while num != 0 {
        digits.push((num % base_u) as i128);
        num /= base_u;
      }
      // digits[i] sits at place base^i; emit highest place first
      let terms: Vec<Expr> = digits
        .iter()
        .enumerate()
        .rev()
        .map(|(place, d)| Expr::Integer(sign * d * base.pow(place as u32)))
        .collect();
      Ok(Expr::List(terms.into()))
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
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "NumberDecompose".to_string(),
    args: args.to_vec().into(),
  };
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
  let value = match to_num(&args[0]) {
    Some(v) => v,
    None => return Ok(unevaluated(args)),
  };
  let unit_exprs = match &args[1] {
    Expr::List(items) if !items.is_empty() => items,
    _ => return Ok(unevaluated(args)),
  };
  let psv = |args: &[Expr]| {
    crate::emit_message(&format!(
      "NumberDecompose::psv: {} is not a list of nonincreasing positive numbers.",
      crate::syntax::expr_to_string(&args[1])
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
    let gcd_i128 = |mut a: i128, mut b: i128| -> i128 {
      a = a.abs();
      b = b.abs();
      while b != 0 {
        (a, b) = (b, a % b);
      }
      a.max(1)
    };
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
        let g = gcd_i128(rn, rd);
        rn /= g;
        rd /= g;
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
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "NumberCompose".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 2 {
    return Ok(unevaluated(args));
  }
  let coeff_exprs = match &args[0] {
    Expr::List(items) => items,
    _ => return Ok(unevaluated(args)),
  };
  let unit_exprs = match &args[1] {
    Expr::List(items) => items,
    _ => return Ok(unevaluated(args)),
  };

  // The coefficient list cannot be longer than the unit list. This is checked
  // before the unit list is validated (matching wolframscript).
  if coeff_exprs.len() > unit_exprs.len() {
    crate::emit_message(&format!(
      "NumberCompose::ulen: List {} of coefficients cannot be longer than list {} of units.",
      crate::syntax::expr_to_string(&args[0]),
      crate::syntax::expr_to_string(&args[1])
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
      crate::syntax::expr_to_string(&args[1])
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
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "MinkowskiQuestionMark".to_string(),
    args: args.to_vec().into(),
  };
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
          op: crate::syntax::UnaryOperator::Minus,
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
          Some((-p, -q, d, -r, true))
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
  let gcd_bigint = |a: &BigInt, b: &BigInt| -> BigInt {
    let (mut a, mut b) = (a.clone(), b.clone());
    if a < BigInt::from(0) {
      a = -a;
    }
    if b < BigInt::from(0) {
      b = -b;
    }
    while b != BigInt::from(0) {
      let r = &a % &b;
      a = b;
      b = r;
    }
    if a == BigInt::from(0) {
      BigInt::from(1)
    } else {
      a
    }
  };
  let reduce = |num: BigInt, den: BigInt| -> (BigInt, BigInt) {
    let g = gcd_bigint(&num, &den);
    let (mut n, mut d) = (num / &g, den / g);
    if d < BigInt::from(0) {
      n = -n;
      d = -d;
    }
    (n, d)
  };
  let add = |a: &(BigInt, BigInt), b: &(BigInt, BigInt)| -> (BigInt, BigInt) {
    reduce(&a.0 * &b.1 + &b.0 * &a.1, &a.1 * &b.1)
  };
  // sign * 2^e as a fraction (e may be negative)
  let pow2 = |sign: i64, e: i128| -> (BigInt, BigInt) {
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
    total = add(&total, &pow2(sign, 1 - s));
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
      t = add(&t, &pow2(cycle_sign, 1 - s - q_sum));
      cycle_sign = -cycle_sign;
    }
    let r_sign: i64 = if period.len() % 2 == 0 { 1 } else { -1 };
    let r = pow2(r_sign, -q_sum);
    let one_minus_r = add(&(BigInt::from(1), BigInt::from(1)), &(-r.0, r.1));
    // tail = t / (1 - r)
    let tail = reduce(&t.0 * &one_minus_r.1, &t.1 * &one_minus_r.0);
    total = add(&total, &tail);
  }

  Ok(super::number_theory::make_rational_expr(total.0, total.1))
}

/// FromRomanNumeral[s] - integer value of a roman numeral string.
/// Case-insensitive over I V X L C D M plus N for zero; values combine
/// with the generic pairwise subtractive rule (a symbol smaller than
/// its successor counts negative), so non-canonical forms like "IIII"
/// (4), "XIIX" (20) and "IM" (999) are accepted. Lists map elementwise.
/// Other characters emit FromRomanNumeral::nrom; non-strings emit
/// FromRomanNumeral::string.
pub fn from_roman_numeral_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "FromRomanNumeral".to_string(),
    args: args.to_vec().into(),
  };
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
      match values {
        Some(v) => {
          let mut total = 0i128;
          for (i, val) in v.iter().enumerate() {
            if v.get(i + 1).is_some_and(|next| val < next) {
              total -= val;
            } else {
              total += val;
            }
          }
          Ok(Expr::Integer(total))
        }
        None => {
          crate::emit_message(&format!(
            "FromRomanNumeral::nrom: String {s} does not represent a valid roman numeral."
          ));
          Ok(unevaluated(args))
        }
      }
    }
    Expr::List(items) => {
      let mut results = Vec::with_capacity(items.len());
      for item in items.iter() {
        results.push(from_roman_numeral_ast(&[item.clone()])?);
      }
      Ok(Expr::List(results.into()))
    }
    _ => {
      crate::emit_message(&format!(
        "FromRomanNumeral::string: String expected at position 1 in {}.",
        crate::syntax::expr_to_string(&unevaluated(args))
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
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "ThueMorse".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 1 {
    return Ok(unevaluated(args));
  }
  match expr_to_bigint(&args[0]) {
    Some(n) => {
      let ones = n
        .abs()
        .to_radix_le(2)
        .1
        .into_iter()
        .filter(|&b| b == 1)
        .count();
      Ok(Expr::Integer((ones % 2) as i128))
    }
    None => {
      // A Real or Rational is numeric but not a valid non-negative integer.
      if matches!(&args[0], Expr::Real(_))
        || matches!(&args[0], Expr::FunctionCall { name, .. } if name == "Rational")
      {
        crate::emit_message(&format!(
          "ThueMorse::nnintprm: Parameter {} at position 1 in {} is expected \
           to be a non-negative integer.",
          crate::syntax::expr_to_string(&args[0]),
          crate::syntax::expr_to_string(&unevaluated(args))
        ));
      }
      Ok(unevaluated(args))
    }
  }
}

/// RudinShapiro[n] - the n-th Rudin–Shapiro number: (-1)^a(n), where a(n) is
/// the number of (possibly overlapping) occurrences of "11" in the binary
/// expansion of n. Defined only for non-negative integers; negative or
/// non-integer arguments stay unevaluated (matching wolframscript).
pub fn rudin_shapiro_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "RudinShapiro".to_string(),
    args: args.to_vec().into(),
  };
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
          crate::syntax::expr_to_string(&args[0]),
          crate::syntax::expr_to_string(&unevaluated(args))
        ));
      }
      Ok(unevaluated(args))
    }
  }
}
