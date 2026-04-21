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
        args: args.to_vec(),
      });
    }
  };
  let base = if args.len() >= 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => b,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "DigitCount".to_string(),
          args: args.to_vec(),
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
          args: args.to_vec(),
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
    Ok(Expr::List(result))
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
        args: args.to_vec(),
      });
    }
  };
  let base = if args.len() == 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => b,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "DigitSum".to_string(),
          args: args.to_vec(),
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
pub fn big_atan_recip(k: u64, terms: usize) -> BigRational {
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
pub fn pi_as_big_rational(terms: usize) -> BigRational {
  let atan5 = big_atan_recip(5, terms);
  let atan239 = big_atan_recip(239, terms);
  // Pi = 4 * (4*atan(1/5) - atan(1/239))
  let four_atan5 = atan5.mul_u64(4);
  let diff = four_atan5.sub(&atan239);
  diff.mul_u64(4)
}

/// Compute E as a BigRational using the series: e = sum(1/k!, k=0..terms)
pub fn e_as_big_rational(terms: usize) -> BigRational {
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
pub fn continued_fraction_from_big_rational(
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
pub fn try_constant_as_big_rational(
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
pub fn continued_fraction_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "ContinuedFraction expects 1 or 2 arguments".into(),
    ));
  }

  // Handle Rational[p, q] or Integer
  match &args[0] {
    Expr::Integer(n) => {
      return Ok(Expr::List(vec![Expr::Integer(*n)]));
    }
    Expr::FunctionCall { name, args: rargs }
      if name == "Rational" && rargs.len() == 2 =>
    {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&rargs[0], &rargs[1]) {
        let mut result = Vec::new();
        let mut a = *p;
        let mut b = *q;
        while b != 0 {
          let quotient = if (a < 0) != (b < 0) && a % b != 0 {
            a / b - 1
          } else {
            a / b
          };
          result.push(Expr::Integer(quotient));
          let rem = a - quotient * b;
          a = b;
          b = rem;
        }
        return Ok(Expr::List(result));
      }
    }
    _ => {}
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
      return Ok(Expr::List(vec![Expr::Integer(a0)]));
    }
    let period_list =
      Expr::List(period.into_iter().map(Expr::Integer).collect());
    return Ok(Expr::List(vec![Expr::Integer(a0), period_list]));
  }

  // For expressions with n terms
  if args.len() == 2 {
    let n = match &args[1] {
      Expr::Integer(n) if *n > 0 => *n as usize,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "ContinuedFraction".to_string(),
          args: args.to_vec(),
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
      return Ok(Expr::List(result));
    }
  }

  Ok(Expr::FunctionCall {
    name: "ContinuedFraction".to_string(),
    args: args.to_vec(),
  })
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
        args: args.to_vec(),
      });
    }
  };

  if elements.is_empty() {
    return Ok(Expr::Identifier("Infinity".to_string()));
  }

  // Collect all integers
  let mut ints: Vec<i128> = Vec::new();
  for elem in elements {
    match elem {
      Expr::Integer(n) => ints.push(*n),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "FromContinuedFraction".to_string(),
          args: args.to_vec(),
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
      args: vec![Expr::Integer(num), Expr::Integer(den)],
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

  let n = match expr_to_bigint(&args[0]) {
    Some(n) => n.abs(),
    None => {
      return Ok(Expr::FunctionCall {
        name: "IntegerDigits".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let base = if args.len() >= 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => BigInt::from(b),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "IntegerDigits: base must be an integer >= 2".into(),
        ));
      }
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
  if args.len() == 3 {
    match expr_to_i128(&args[2]) {
      Some(len) if len >= 0 => {
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
      _ => {
        return Err(InterpreterError::EvaluationError(
          "IntegerDigits: length must be a non-negative integer".into(),
        ));
      }
    }
  }

  Ok(Expr::List(digits))
}

/// Extract decimal digits and exponent from a BigFloat.
/// Returns (digit_chars, decimal_exponent) where digit_chars are ASCII digit bytes
/// and decimal_exponent is the number of integer digits (position of decimal point).
pub fn bigfloat_to_digits(
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

/// Convert a decimal-literal Real to an exact rational `(numerator, denominator)`
/// Compute RealDigits for a rational number using long division with cycle detection.
/// Returns the digit list (with repeating part wrapped in a List) and the exponent.
fn real_digits_rational(numer: i128, denom: i128) -> (Vec<Expr>, i128) {
  real_digits_rational_base(numer, denom, 10)
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
    result.push(Expr::List(repeating));
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
    // For machine-precision Reals in a non-decimal base, produce roughly
    // as many significant digits as 16 decimal digits would take in the
    // target base: ceil(16 * log10(10) / log10(base)).
    let ratio = 16.0_f64 / (base as f64).log10();
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
      args: vec![expr.clone()],
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
    return Ok(Expr::List(vec![Expr::List(digits), Expr::Integer(exp)]));
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
      return Ok(Expr::List(vec![
        Expr::List(digit_list),
        Expr::Integer(exponent),
      ]));
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

      // Determine exponent from integer part
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

      // Machine-precision cap: a Real input carries only ~16 significant
      // decimal digits of information. Any digit beyond that is unknowable
      // and wolframscript marks it `Indeterminate`.
      let precision_cap = if rational_from_real.is_some() {
        Some(16usize)
      } else {
        None
      };
      let max_known = precision_cap.unwrap_or(num_digits).min(num_digits);

      // Generate up to `max_known` known digits.
      while digits.len() < max_known {
        remainder *= base;
        let digit = remainder / d;
        remainder %= d;
        digits.push(digit);
      }
      digits.truncate(max_known);

      let mut digit_exprs: Vec<Expr> =
        digits.iter().map(|&dig| Expr::Integer(dig)).collect();
      // Pad beyond machine precision with Indeterminate.
      while digit_exprs.len() < num_digits {
        digit_exprs.push(Expr::Identifier("Indeterminate".to_string()));
      }
      return Ok(Expr::List(vec![
        Expr::List(digit_exprs),
        Expr::Integer(exponent),
      ]));
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
        args: args.to_vec(),
      });
    }
  };

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
    return Ok(Expr::List(vec![
      Expr::List(digit_exprs),
      Expr::Integer(p + 1),
    ]));
  }

  // Pad with zeros if we don't have enough digits
  while digits.len() < num_digits {
    digits.push(0);
  }

  // Truncate to requested number of digits
  digits.truncate(num_digits);

  let digit_exprs: Vec<Expr> =
    digits.iter().map(|&d| Expr::Integer(d)).collect();

  Ok(Expr::List(vec![
    Expr::List(digit_exprs),
    Expr::Integer(decimal_exp as i128),
  ]))
}

/// FromDigits[list
pub fn from_digits_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "FromDigits expects 1 or 2 arguments".into(),
    ));
  }

  // The base may be an integer >= 2 (fast numeric path) or a symbolic
  // expression (e.g. `FromDigits[{1,2,3}, x]` => `3 + 2*x + x^2`).
  let numeric_base: Option<i128> = if args.len() == 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => Some(b),
      Some(_) => {
        return Err(InterpreterError::EvaluationError(
          "FromDigits: base must be an integer >= 2".into(),
        ));
      }
      None => None,
    }
  } else {
    Some(10)
  };

  // Symbolic base path: build an expanded polynomial sum in the base.
  // For `FromDigits[{d0, d1, ..., d_{n-1}}, b]` we produce
  //   d0*b^(n-1) + d1*b^(n-2) + ... + d_{n-2}*b + d_{n-1}.
  if numeric_base.is_none() {
    let items = match &args[0] {
      Expr::List(items) => items.clone(),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "FromDigits".to_string(),
          args: args.to_vec(),
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
            args: vec![base_expr.clone(), Expr::Integer(power)],
          }
        };
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![item.clone(), base_pow],
        }
      };
      terms.push(term);
    }
    let sum = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms,
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
          args: args.to_vec(),
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
        args: args.to_vec(),
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
      fn bigint_gcd(a: &BigInt, b: &BigInt) -> BigInt {
        let mut a = if a < &BigInt::from(0) { -a } else { a.clone() };
        let mut b = if b < &BigInt::from(0) { -b } else { b.clone() };
        while b > BigInt::from(0) {
          let t = &a % &b;
          a = b;
          b = t;
        }
        a
      }
      let g = bigint_gcd(&int_val, &denom);
      let num = &int_val / &g;
      let den = &denom / &g;
      if den == BigInt::from(1) {
        return Ok(bigint_to_expr(num));
      } else {
        return Ok(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![bigint_to_expr(num), bigint_to_expr(den)],
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
            args: vec![base_expr.clone(), result],
          },
          item.clone(),
        ],
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
          args: args.to_vec(),
        });
      }
      None => {
        return Ok(Expr::FunctionCall {
          name: "IntegerLength".to_string(),
          args: args.to_vec(),
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
    args: args.to_vec(),
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
          args: args.to_vec(),
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
    args: args.to_vec(),
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
pub fn spell_below_1000(n: u64) -> String {
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

pub fn integer_name_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // IntegerName[n] - convert integer to English name
  // IntegerName also works on lists
  if args.len() == 1
    && let Expr::List(items) = &args[0]
  {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| integer_name_ast(&[item.clone()]))
      .collect();
    return Ok(Expr::List(results?));
  }

  let n = match expr_to_i128(&args[0]) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: "IntegerName".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let negative = n < 0;
  let abs_n = n.unsigned_abs();

  if abs_n == 0 {
    return Ok(Expr::String("zero".to_string()));
  }

  // For numbers 1..=999, spell out entirely in words
  if abs_n <= 999 {
    let word = spell_below_1000(abs_n as u64);
    let result = if negative {
      format!("negative {}", word)
    } else {
      word
    };
    return Ok(Expr::String(result));
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
        return Ok(Expr::FunctionCall {
          name: "IntegerName".to_string(),
          args: args.to_vec(),
        });
      };
      parts.push(format!("{} {}", group, scale));
    }
  }

  let result = parts.join(" ");
  let result = if negative {
    format!("negative {}", result)
  } else {
    result
  };
  Ok(Expr::String(result))
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
        args: args.to_vec(),
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
      args: args.to_vec(),
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
        args: args.to_vec(),
      });
    }
  };

  if elements.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  // Collect all integers
  let mut ints: Vec<i128> = Vec::new();
  for elem in elements {
    match elem {
      Expr::Integer(n) => ints.push(*n),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Convergents".to_string(),
          args: args.to_vec(),
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

  Ok(Expr::List(result))
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
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "NumberDigit expects exactly 2 arguments".into(),
    ));
  }

  let val = match &args[0] {
    Expr::Integer(n) => *n as f64,
    Expr::Real(f) => *f,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "NumberDigit".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let pos = match expr_to_i128(&args[1]) {
    Some(n) => n,
    None => {
      return Err(InterpreterError::EvaluationError(
        "NumberDigit: position must be an integer".into(),
      ));
    }
  };

  let abs_val = val.abs();

  // Shift the number so the desired digit is in the ones place,
  // then extract it: floor(abs_val / 10^pos) % 10
  let factor = 10f64.powi(pos as i32);
  let digit = ((abs_val / factor).floor() as i128) % 10;

  Ok(Expr::Integer(digit))
}
