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
    BigUint { digits: vec![0] }
  }

  fn from_u64(n: u64) -> Self {
    BigUint { digits: vec![n] }
  }

  fn from_u128(n: u128) -> Self {
    let lo = n as u64;
    let hi = (n >> 64) as u64;
    let mut b = BigUint {
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

  fn cmp(&self, other: &BigUint) -> std::cmp::Ordering {
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
  fn add(&self, other: &BigUint) -> BigUint {
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
    let mut r = BigUint { digits: result };
    r.trim();
    r
  }

  /// self - other (assumes self >= other)
  fn sub(&self, other: &BigUint) -> BigUint {
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
    let mut r = BigUint { digits: result };
    r.trim();
    r
  }

  /// self * other
  fn mul(&self, other: &BigUint) -> BigUint {
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
    let mut r = BigUint { digits: result };
    r.trim();
    r
  }

  /// self * scalar
  fn mul_u64(&self, scalar: u64) -> BigUint {
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
    let mut r = BigUint { digits: result };
    r.trim();
    r
  }

  /// Division: returns (quotient, remainder)
  fn divmod(&self, other: &BigUint) -> (BigUint, BigUint) {
    use std::cmp::Ordering;
    if other.is_zero() {
      panic!("BigUint division by zero");
    }
    match self.cmp(other) {
      Ordering::Less => return (BigUint::zero(), self.clone()),
      Ordering::Equal => return (BigUint::from_u64(1), BigUint::zero()),
      _ => {}
    }
    if other.digits.len() == 1 {
      let d = other.digits[0];
      let (q, r) = self.divmod_u64(d);
      return (q, BigUint::from_u64(r));
    }
    // Long division
    self.long_divmod(other)
  }

  /// Divide by a single u64, returns (quotient, remainder)
  fn divmod_u64(&self, d: u64) -> (BigUint, u64) {
    let mut result = vec![0u64; self.digits.len()];
    let mut rem: u128 = 0;
    for i in (0..self.digits.len()).rev() {
      rem = (rem << 64) | (self.digits[i] as u128);
      result[i] = (rem / d as u128) as u64;
      rem %= d as u128;
    }
    let mut q = BigUint { digits: result };
    q.trim();
    (q, rem as u64)
  }

  /// Long division for multi-digit divisors
  fn long_divmod(&self, other: &BigUint) -> (BigUint, BigUint) {
    // Shift-and-subtract algorithm operating on bits
    let mut remainder = BigUint::zero();
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
    let mut q = BigUint {
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

  fn shl1(&self) -> BigUint {
    let mut result = Vec::with_capacity(self.digits.len() + 1);
    let mut carry = 0u64;
    for &d in &self.digits {
      result.push((d << 1) | carry);
      carry = d >> 63;
    }
    if carry > 0 {
      result.push(carry);
    }
    let mut r = BigUint { digits: result };
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

  fn gcd(a: &BigUint, b: &BigUint) -> BigUint {
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
    BigRational {
      num: BigUint::zero(),
      den: BigUint::from_u64(1),
      negative: false,
    }
  }

  fn from_i64(n: i64) -> Self {
    BigRational {
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
  fn add(&self, other: &BigRational) -> BigRational {
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
    let mut r = BigRational {
      num,
      den: bd,
      negative,
    };
    r.reduce();
    r
  }

  /// self - other
  fn sub(&self, other: &BigRational) -> BigRational {
    let neg_other = BigRational {
      num: other.num.clone(),
      den: other.den.clone(),
      negative: !other.negative,
    };
    self.add(&neg_other)
  }

  /// self * scalar (positive integer)
  fn mul_u64(&self, s: u64) -> BigRational {
    let mut r = BigRational {
      num: self.num.mul_u64(s),
      den: self.den.clone(),
      negative: self.negative,
    };
    r.reduce();
    r
  }

  /// Floor division: returns (floor, remainder) such that self = floor + remainder/1
  /// where 0 <= remainder < 1 (for positive self)
  fn floor_and_remainder(&self) -> (i128, BigRational) {
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
    let floor_rat = BigRational::from_i64(floor_val as i64);
    let rem = self.sub(&floor_rat);
    (floor_val, rem)
  }

  /// 1 / self
  fn reciprocal(&self) -> BigRational {
    BigRational {
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
) -> Result<(Vec<u8>, i64), InterpreterError> {
  use num_bigint::BigUint;
  use num_traits::Zero;

  let (words, sig_bits, _sign, exponent, _inexact) =
    bf.as_raw_parts().ok_or_else(|| {
      InterpreterError::EvaluationError(
        "RealDigits: cannot extract NaN or Inf".into(),
      )
    })?;

  if sig_bits == 0 || words.iter().all(|&w| w == 0) {
    return Ok((vec![b'0'], 0));
  }

  let mantissa = BigUint::from_bytes_le(
    &words
      .iter()
      .flat_map(|w| w.to_le_bytes())
      .collect::<Vec<u8>>(),
  );

  let mantissa_bits = words.len() * 64;
  let shift = exponent as i64 - mantissa_bits as i64;

  let target_digits =
    (mantissa_bits as f64 / std::f64::consts::LOG2_10).ceil() as usize + 2;

  let (int_digits, decimal_exp) = if shift >= 0 {
    let int_val = &mantissa << (shift as u64);
    let s = int_val.to_string();
    let len = s.len();
    (s, len as i64)
  } else {
    let neg_shift = (-shift) as u64;
    let scale = BigUint::from(10u32).pow(target_digits as u32);
    let scaled = &mantissa * &scale;
    let result = &scaled >> neg_shift;

    if result.is_zero() {
      return Ok((vec![b'0'], 0));
    }

    let s = result.to_string();
    let decimal_exp = s.len() as i64 - target_digits as i64;
    (s, decimal_exp)
  };

  Ok((int_digits.into_bytes(), decimal_exp))
}

/// RealDigits[x, base, num_digits] — extract decimal digits of a real number.
/// Returns {digit_list, exponent}.
pub fn real_digits_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "RealDigits expects 1 to 3 arguments".into(),
    ));
  }

  // Only base 10 is supported for now
  if args.len() >= 2 {
    match expr_to_i128(&args[1]) {
      Some(10) => {}
      Some(_) => {
        return Ok(Expr::FunctionCall {
          name: "RealDigits".to_string(),
          args: args.to_vec(),
        });
      }
      None => {
        return Err(InterpreterError::EvaluationError(
          "RealDigits: base must be an integer".into(),
        ));
      }
    }
  }

  let num_digits: usize = if args.len() >= 3 {
    match expr_to_i128(&args[2]) {
      Some(n) if n > 0 => n as usize,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "RealDigits: number of digits must be a positive integer".into(),
        ));
      }
    }
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
    let digits = vec![Expr::Integer(0); num_digits];
    return Ok(Expr::List(vec![Expr::List(digits), Expr::Integer(0)]));
  }

  // Compute with extra precision to avoid rounding errors in the last digits
  let extra = 10;
  let precision = num_digits + extra;

  use astro_float::{Consts, RoundingMode};
  let mut cc = Consts::new().map_err(|e| {
    InterpreterError::EvaluationError(format!("BigFloat init error: {}", e))
  })?;
  let rm = RoundingMode::ToEven;
  let bits = nominal_bits(precision);

  let bf = expr_to_bigfloat(&abs_expr, bits, rm, &mut cc)?;

  let (raw_digits, decimal_exp) = bigfloat_to_digits(&bf)?;

  // raw_digits are the significant digits, decimal_exp is the exponent
  // (number of digits before the decimal point).
  // We need exactly num_digits digits.
  let mut digits: Vec<i128> = raw_digits
    .iter()
    .filter(|b| b.is_ascii_digit())
    .map(|b| (*b - b'0') as i128)
    .collect();

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

  let base: i128 = if args.len() == 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => b,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "FromDigits: base must be an integer >= 2".into(),
        ));
      }
    }
  } else {
    10
  };

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
      if d >= base {
        return Err(InterpreterError::EvaluationError(format!(
          "FromDigits: invalid digit {} for base {}",
          ch, base
        )));
      }
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
        eprintln!();
        eprintln!(
          "IntegerLength::ibase: Base {} is not an integer greater than 1.",
          b
        );
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
    return Ok(Expr::Identifier("N".to_string()));
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

  Ok(Expr::Identifier(result))
}
