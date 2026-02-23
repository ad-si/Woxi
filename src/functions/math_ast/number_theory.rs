#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;
use num_bigint::BigInt;
use num_traits::Signed;

pub fn bigint_gcd(a: BigInt, b: BigInt) -> BigInt {
  use num_traits::Zero;
  let (mut a, mut b) = (a.abs(), b.abs());
  while !b.is_zero() {
    let t = b.clone();
    b = &a % &b;
    a = t;
  }
  a
}

/// GCD[a, b, ...] - Greatest common divisor
pub fn gcd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Integer(0));
  }

  let mut result: Option<BigInt> = None;
  for arg in args {
    let val = match arg {
      Expr::Integer(n) => BigInt::from(*n),
      Expr::BigInteger(n) => n.clone(),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "GCD".to_string(),
          args: args.to_vec(),
        });
      }
    };
    result = Some(match result {
      Some(r) => bigint_gcd(r, val),
      None => val.abs(),
    });
  }

  Ok(bigint_to_expr(result.unwrap_or_else(|| BigInt::from(0))))
}

/// Extended Euclidean algorithm: returns (gcd, s, t) where a*s + b*t = gcd
pub fn extended_gcd_bigint(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
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
        return Ok(Expr::FunctionCall {
          name: "ExtendedGCD".to_string(),
          args: args.to_vec(),
        });
      }
    };
    vals.push(val);
  }

  if vals.len() == 2 {
    let (g, s, t) = extended_gcd_bigint(&vals[0], &vals[1]);
    return Ok(Expr::List(vec![
      bigint_to_expr(g),
      Expr::List(vec![bigint_to_expr(s), bigint_to_expr(t)]),
    ]));
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

  Ok(Expr::List(vec![
    bigint_to_expr(g),
    Expr::List(coeffs.into_iter().map(bigint_to_expr).collect()),
  ]))
}

/// LCM[a, b, ...] - Least common multiple
pub fn lcm_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Integer(1));
  }

  let mut result: Option<BigInt> = None;
  for arg in args {
    let val = match arg {
      Expr::Integer(n) => BigInt::from(*n),
      Expr::BigInteger(n) => n.clone(),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "LCM".to_string(),
          args: args.to_vec(),
        });
      }
    };
    result = Some(match result {
      Some(r) => {
        use num_traits::Zero;
        if r.is_zero() || val.is_zero() {
          BigInt::from(0)
        } else {
          let g = bigint_gcd(r.clone(), val.clone());
          (r.abs() / g) * val.abs()
        }
      }
      None => val.abs(),
    });
  }

  Ok(bigint_to_expr(result.unwrap_or_else(|| BigInt::from(1))))
}

/// Helper function to compute GCD
pub fn gcd_helper(a: i128, b: i128) -> i128 {
  if b == 0 { a } else { gcd_helper(b, a % b) }
}

/// Factorial[n] or n!
pub fn factorial_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Factorial expects exactly 1 argument".into(),
    ));
  }
  if let Some(n) = expr_to_i128(&args[0]) {
    if n < 0 {
      return Err(InterpreterError::EvaluationError(
        "Factorial: argument must be non-negative".into(),
      ));
    }
    let mut result = BigInt::from(1);
    for i in 2..=n {
      result *= i;
    }
    Ok(bigint_to_expr(result))
  } else {
    Ok(Expr::FunctionCall {
      name: "Factorial".to_string(),
      args: args.to_vec(),
    })
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
        // Compute by working from -1 down
        let mut numer: i128 = 1;
        let mut denom: i128 = 1;
        let mut k = -1i128;
        while k > n {
          k -= 2;
          // (k)!! = (k+2)!! / (k+2)
          denom *= k + 2;
          // Simplify
          let g = gcd_i128(numer.abs(), denom.abs());
          numer /= g;
          denom /= g;
        }
        if denom < 0 {
          numer = -numer;
          denom = -denom;
        }
        if denom == 1 {
          return Ok(Expr::Integer(numer));
        }
        return Ok(make_rational(numer, denom));
      }
      return Ok(Expr::FunctionCall {
        name: "Factorial2".to_string(),
        args: args.to_vec(),
      });
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
  } else {
    Ok(Expr::FunctionCall {
      name: "Factorial2".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Subfactorial[n] - Count of derangements: !n = n! * Sum[(-1)^k/k!, {k, 0, n}]
pub fn subfactorial_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Subfactorial expects exactly 1 argument".into(),
    ));
  }
  if let Some(n) = expr_to_i128(&args[0]) {
    if n < 0 {
      return Ok(Expr::FunctionCall {
        name: "Subfactorial".to_string(),
        args: args.to_vec(),
      });
    }
    if n == 0 {
      return Ok(Expr::Integer(1));
    }
    // Use recurrence: !n = (n-1) * (!(n-1) + !(n-2))
    let mut prev2 = BigInt::from(1); // !0 = 1
    let mut prev1 = BigInt::from(0); // !1 = 0
    if n == 1 {
      return Ok(Expr::Integer(0));
    }
    for i in 2..=n {
      let current = BigInt::from(i - 1) * (&prev1 + &prev2);
      prev2 = prev1;
      prev1 = current;
    }
    Ok(bigint_to_expr(prev1))
  } else {
    Ok(Expr::FunctionCall {
      name: "Subfactorial".to_string(),
      args: args.to_vec(),
    })
  }
}

/// LucasL[n] - Lucas number L_n
pub fn lucas_l_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "LucasL expects exactly 1 argument".into(),
    ));
  }
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "LucasL".to_string(),
        args: args.to_vec(),
      });
    }
  };
  // L(0) = 2, L(1) = 1, L(n) = L(n-1) + L(n-2)
  if n == 0 {
    return Ok(Expr::Integer(2));
  }
  if n == 1 {
    return Ok(Expr::Integer(1));
  }
  let mut a = BigInt::from(2);
  let mut b = BigInt::from(1);
  for _ in 2..=n {
    let c = &a + &b;
    a = b;
    b = c;
  }
  Ok(bigint_to_expr(b))
}

/// ChineseRemainder[{r1,r2,...}, {m1,m2,...}] - Chinese Remainder Theorem
pub fn chinese_remainder_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ChineseRemainder expects exactly 2 arguments".into(),
    ));
  }
  let remainders = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ChineseRemainder".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let moduli = match &args[1] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ChineseRemainder".to_string(),
        args: args.to_vec(),
      });
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
          return Ok(Expr::FunctionCall {
            name: "ChineseRemainder".to_string(),
            args: args.to_vec(),
          });
        }
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "ChineseRemainder".to_string(),
          args: args.to_vec(),
        });
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
      return Err(InterpreterError::EvaluationError(
        "ChineseRemainder: no solution exists".into(),
      ));
    }
    let lcm = modulus / g * mi;
    result =
      (result + modulus * ((ri - result) / g % (mi / g)) * p).rem_euclid(lcm);
    modulus = lcm;
  }

  Ok(Expr::Integer(result))
}

/// DivisorSum[n, form] - applies form to each divisor and sums
pub fn divisor_sum_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "DivisorSum expects exactly 2 arguments".into(),
    ));
  }
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n > 0 => n,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DivisorSum".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let func = &args[1];

  // Get divisors
  let mut divs = Vec::new();
  for i in 1..=n {
    if n % i == 0 {
      divs.push(i);
    }
  }

  // Apply function to each divisor and sum
  let mut sum = Expr::Integer(0);
  for d in divs {
    let val = crate::evaluator::apply_function_to_arg(func, &Expr::Integer(d))?;
    sum = super::plus_ast(&[sum, val])?;
  }
  Ok(sum)
}

// ─── Combinatorics Functions ─────────────────────────────────────

/// BernoulliB[n] - nth Bernoulli number
pub fn bernoulli_b_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "BernoulliB expects exactly 1 argument".into(),
    ));
  }
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "BernoulliB".to_string(),
        args: args.to_vec(),
      });
    }
  };

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

  // Represent as (numerator, denominator)
  let mut b: Vec<(i128, i128)> = Vec::with_capacity(n + 1);
  b.push((1, 1)); // B(0) = 1
  if n >= 1 {
    b.push((-1, 2)); // B(1) = -1/2
  }

  fn rat_gcd(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
      let t = b;
      b = a % b;
      a = t;
    }
    a
  }

  fn rat_add(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
    let num = a.0 * b.1 + b.0 * a.1;
    let den = a.1 * b.1;
    let g = rat_gcd(num, den);
    if den < 0 {
      (-num / g, -den / g)
    } else {
      (num / g, den / g)
    }
  }

  fn rat_mul(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
    let num = a.0 * b.0;
    let den = a.1 * b.1;
    let g = rat_gcd(num, den);
    if den < 0 {
      (-num / g, -den / g)
    } else {
      (num / g, den / g)
    }
  }

  for m in 2..=n {
    if m % 2 != 0 && m > 1 {
      b.push((0, 1));
      continue;
    }
    // B(m) = -1/(m+1) * sum_{k=0}^{m-1} C(m+1, k) * B(k)
    let mut sum: (i128, i128) = (0, 1);
    let mut binom: i128 = 1; // C(m+1, k) starting at k=0
    for k in 0..m {
      sum = rat_add(sum, rat_mul((binom, 1), b[k]));
      binom = binom * (m as i128 + 1 - k as i128) / (k as i128 + 1);
    }
    let result = rat_mul((-1, m as i128 + 1), sum);
    b.push(result);
  }

  let (num, den) = b[n];
  Ok(make_rational(num, den))
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
      return Ok(Expr::FunctionCall {
        name: "BellB".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if args.len() == 1 {
    // Bell number B_n via the Bell triangle
    if n == 0 {
      return Ok(Expr::Integer(1));
    }
    // Compute using the Bell triangle (first column of each row)
    let mut row = vec![1i128];
    for _ in 1..=n {
      let mut new_row = vec![*row.last().unwrap()];
      for j in 1..=row.len() {
        let val = new_row[j - 1] + row[j - 1];
        new_row.push(val);
      }
      row = new_row;
    }
    Ok(Expr::Integer(row[0]))
  } else {
    // Bell polynomial B_n(x) = sum_{k=0}^{n} S(n,k) * x^k
    // where S(n,k) is the Stirling number of the second kind
    if n == 0 {
      return Ok(Expr::Integer(1));
    }
    // Compute Stirling numbers of the second kind for all k
    let mut stirling = vec![vec![0i128; n + 1]; n + 1];
    stirling[0][0] = 1;
    for i in 1..=n {
      for k in 1..=i {
        stirling[i][k] =
          k as i128 * stirling[i - 1][k] + stirling[i - 1][k - 1];
      }
    }
    // Build polynomial: sum_{k=0}^{n} S(n,k) * x^k
    let x = &args[1];
    let mut terms = Vec::new();
    for k in 0..=n {
      let s = stirling[n][k];
      if s == 0 {
        continue;
      }
      let coeff = Expr::Integer(s);
      let term = if k == 0 {
        coeff
      } else if k == 1 {
        if s == 1 {
          x.clone()
        } else {
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(coeff),
            right: Box::new(x.clone()),
          }
        }
      } else {
        let power = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Power,
          left: Box::new(x.clone()),
          right: Box::new(Expr::Integer(k as i128)),
        };
        if s == 1 {
          power
        } else {
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
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
        op: crate::syntax::BinaryOperator::Plus,
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
      return Ok(Expr::FunctionCall {
        name: "PauliMatrix".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let i_expr = Expr::Identifier("I".to_string());
  let neg_i = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Times,
    left: Box::new(Expr::Integer(-1)),
    right: Box::new(i_expr.clone()),
  };
  match k {
    0 => Ok(Expr::List(vec![
      Expr::List(vec![Expr::Integer(1), Expr::Integer(0)]),
      Expr::List(vec![Expr::Integer(0), Expr::Integer(1)]),
    ])),
    1 => Ok(Expr::List(vec![
      Expr::List(vec![Expr::Integer(0), Expr::Integer(1)]),
      Expr::List(vec![Expr::Integer(1), Expr::Integer(0)]),
    ])),
    2 => {
      let neg_i_eval = crate::evaluator::evaluate_expr_to_expr(&neg_i)?;
      Ok(Expr::List(vec![
        Expr::List(vec![Expr::Integer(0), neg_i_eval]),
        Expr::List(vec![i_expr, Expr::Integer(0)]),
      ]))
    }
    3 => Ok(Expr::List(vec![
      Expr::List(vec![Expr::Integer(1), Expr::Integer(0)]),
      Expr::List(vec![Expr::Integer(0), Expr::Integer(-1)]),
    ])),
    _ => Ok(Expr::FunctionCall {
      name: "PauliMatrix".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// CatalanNumber[n] - nth Catalan number = C(2n,n)/(n+1)
pub fn catalan_number_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "CatalanNumber expects exactly 1 argument".into(),
    ));
  }
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "CatalanNumber".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // C(2n, n) / (n + 1)
  let mut result: i128 = 1;
  for i in 0..n {
    result = result * (2 * n - i) / (i + 1);
  }
  result /= n + 1;
  Ok(Expr::Integer(result))
}

/// StirlingS1[n, k] - Stirling number of the first kind (signed)
pub fn stirling_s1_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StirlingS1 expects exactly 2 arguments".into(),
    ));
  }
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "StirlingS1".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let k = match expr_to_i128(&args[1]) {
    Some(k) if k >= 0 => k as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "StirlingS1".to_string(),
        args: args.to_vec(),
      });
    }
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
  let mut table = vec![vec![zero.clone(); k + 1]; n + 1];
  table[0][0] = one;
  for i in 1..=n {
    for j in 1..=k.min(i) {
      table[i][j] =
        &table[i - 1][j - 1] - BigInt::from(i - 1) * &table[i - 1][j];
    }
  }
  Ok(bigint_to_expr(table[n][k].clone()))
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
    _ => {
      return Ok(Expr::FunctionCall {
        name: "StirlingS2".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let k = match expr_to_i128(&args[1]) {
    Some(k) if k >= 0 => k as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "StirlingS2".to_string(),
        args: args.to_vec(),
      });
    }
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
  let mut table = vec![vec![zero.clone(); k + 1]; n + 1];
  table[0][0] = one;
  for i in 1..=n {
    for j in 1..=k.min(i) {
      table[i][j] = BigInt::from(j) * &table[i - 1][j] + &table[i - 1][j - 1];
    }
  }
  Ok(bigint_to_expr(table[n][k].clone()))
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
    1.0 / 12.0,      // B_2/2 = 1/12
    1.0 / 120.0,     // B_4/4 = -1/30 / 4 => but sign alternates
    1.0 / 252.0,     // B_6/6
    1.0 / 240.0,     // B_8/8
    5.0 / 660.0,     // B_10/10
    691.0 / 32760.0, // B_12/12
    7.0 / 12.0,      // B_14/14
  ];
  // Signs: -, +, -, +, -, +, -
  let signs = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
  for i in 0..coeffs.len() {
    result += signs[i] * coeffs[i] / xpow;
    xpow *= x2;
  }
  result
}

/// HarmonicNumber[n] - Returns the nth harmonic number H_n = 1 + 1/2 + ... + 1/n.
/// HarmonicNumber[n, r] - Returns the generalized harmonic number H_{n,r} = Sum[1/k^r, {k,1,n}].
pub fn harmonic_number_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "HarmonicNumber expects 1 or 2 arguments".into(),
    ));
  }

  // Handle real/float argument: H(x) = digamma(x+1) + EulerGamma
  if args.len() == 1
    && let Some(x) = expr_to_num(&args[0])
    && expr_to_i128(&args[0]).is_none()
  {
    // Real input - use digamma approximation
    // Euler-Mascheroni constant
    const EULER_GAMMA: f64 = 0.5772156649015329;
    let result = digamma(x + 1.0) + EULER_GAMMA;
    return Ok(Expr::Real(result));
  }

  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n,
    Some(_) => {
      return Ok(Expr::FunctionCall {
        name: "HarmonicNumber".to_string(),
        args: args.to_vec(),
      });
    }
    None => {
      return Ok(Expr::FunctionCall {
        name: "HarmonicNumber".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let r = if args.len() == 2 {
    match expr_to_i128(&args[1]) {
      Some(r) => r,
      None => {
        return Ok(Expr::FunctionCall {
          name: "HarmonicNumber".to_string(),
          args: args.to_vec(),
        });
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
  fn bigint_gcd(a: &BigInt, b: &BigInt) -> BigInt {
    use num_traits::Zero;
    let mut a = if *a < BigInt::zero() {
      -a.clone()
    } else {
      a.clone()
    };
    let mut b = if *b < BigInt::zero() {
      -b.clone()
    } else {
      b.clone()
    };
    while !b.is_zero() {
      let t = b.clone();
      b = &a % &b;
      a = t;
    }
    a
  }

  let mut num = BigInt::from(0);
  let mut den = BigInt::from(1);
  for k in 1..=n {
    let k_big = BigInt::from(k);
    let k_pow = num_traits::pow::pow(k_big, r as usize);
    // Add 1/k_pow to num/den: num/den + 1/k_pow = (num*k_pow + den) / (den*k_pow)
    num = &num * &k_pow + &den;
    den = &den * &k_pow;
    // Reduce
    let g = bigint_gcd(&num, &den);
    use num_traits::One;
    if g > BigInt::one() {
      num /= &g;
      den /= &g;
    }
  }

  // Convert to our representation
  if den == BigInt::from(1) {
    Ok(bigint_to_expr(num))
  } else {
    Ok(Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![bigint_to_expr(num), bigint_to_expr(den)],
    })
  }
}

/// Prime[n] - Returns the nth prime number
pub fn prime_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Prime expects exactly 1 argument".into(),
    ));
  }
  match expr_to_i128(&args[0]).or_else(|| {
    if let Expr::Real(f) = &args[0] {
      if f.fract() == 0.0 {
        Some(*f as i128)
      } else {
        None
      }
    } else {
      None
    }
  }) {
    Some(n) if n >= 1 => {
      Ok(Expr::Integer(crate::nth_prime(n as usize) as i128))
    }
    _ => {
      // Wolfram returns unevaluated for non-positive or non-integer arguments
      Ok(Expr::FunctionCall {
        name: "Prime".to_string(),
        args: args.to_vec(),
      })
    }
  }
}

/// Fibonacci[n] - Returns the nth Fibonacci number
pub fn fibonacci_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
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
    None => Ok(Expr::FunctionCall {
      name: "Fibonacci".to_string(),
      args: args.to_vec(),
    }),
  }
}

pub fn fibonacci_number_bigint(n: u128) -> BigInt {
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
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "FactorInteger expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    Expr::Integer(n) => factor_integer_i128(*n),
    Expr::BigInteger(n) => factor_integer_bigint(n),
    _ => Ok(Expr::FunctionCall {
      name: "FactorInteger".to_string(),
      args: args.to_vec(),
    }),
  }
}

pub fn factor_integer_i128(n: i128) -> Result<Expr, InterpreterError> {
  if n == 0 {
    return Err(InterpreterError::EvaluationError(
      "FactorInteger: argument cannot be zero".into(),
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
    factors.push(Expr::List(vec![Expr::Integer(-1), Expr::Integer(1)]));
  }

  if num == 1 {
    return Ok(Expr::List(factors));
  }

  // Handle factor of 2
  let mut count = 0i128;
  while num.is_multiple_of(2) {
    count += 1;
    num /= 2;
  }
  if count > 0 {
    factors.push(Expr::List(vec![Expr::Integer(2), Expr::Integer(count)]));
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
      factors.push(Expr::List(vec![
        Expr::Integer(i as i128),
        Expr::Integer(count),
      ]));
    }
    i += 2;
  }

  if num > 1 {
    factors.push(Expr::List(vec![
      Expr::Integer(num as i128),
      Expr::Integer(1),
    ]));
  }

  Ok(Expr::List(factors))
}

pub fn factor_integer_bigint(n: &BigInt) -> Result<Expr, InterpreterError> {
  use num_traits::{One, Signed, Zero};

  if n.is_zero() {
    return Err(InterpreterError::EvaluationError(
      "FactorInteger: argument cannot be zero".into(),
    ));
  }

  let mut factors: Vec<Expr> = Vec::new();

  if n.is_negative() {
    factors.push(Expr::List(vec![Expr::Integer(-1), Expr::Integer(1)]));
  }

  let mut remaining = n.abs();
  let one = BigInt::one();

  if remaining == one {
    return Ok(Expr::List(factors));
  }

  // Trial division for small primes
  let two = BigInt::from(2);
  let mut count = 0i128;
  while (&remaining % &two).is_zero() {
    count += 1;
    remaining /= &two;
  }
  if count > 0 {
    factors.push(Expr::List(vec![Expr::Integer(2), Expr::Integer(count)]));
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
      factors.push(Expr::List(vec![
        Expr::Integer(i as i128),
        Expr::Integer(count),
      ]));
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
        factors.push(Expr::List(vec![
          bigint_to_expr(factor_bigint),
          Expr::Integer(exponent as i128),
        ]));
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

  Ok(Expr::List(factors))
}

/// Divisors[n] - Returns a sorted list of all divisors of n
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
      return Ok(Expr::FunctionCall {
        name: "IntegerPartitions".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if n < 0 {
    return Ok(Expr::List(vec![]));
  }
  let n = n as u64;

  // Parse length constraints from second arg
  let (min_len, max_len) = if args.len() >= 2 {
    match &args[1] {
      // IntegerPartitions[n, k] — at most k parts
      e if expr_to_i128(e).is_some_and(|k| k >= 0) => {
        (1, expr_to_i128(e).unwrap() as u64)
      }
      // IntegerPartitions[n, All] — no constraint
      Expr::Identifier(s) if s == "All" => (1, n.max(1)),
      // IntegerPartitions[n, {k}] — exactly k parts
      Expr::List(lst) if lst.len() == 1 => match expr_to_i128(&lst[0]) {
        Some(k) if k >= 0 => (k as u64, k as u64),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "IntegerPartitions".to_string(),
            args: args.to_vec(),
          });
        }
      },
      // IntegerPartitions[n, {kmin, kmax}] — range of parts
      Expr::List(lst) if lst.len() == 2 => {
        match (expr_to_i128(&lst[0]), expr_to_i128(&lst[1])) {
          (Some(lo), Some(hi)) if lo >= 0 && hi >= 0 => (lo as u64, hi as u64),
          _ => {
            return Ok(Expr::FunctionCall {
              name: "IntegerPartitions".to_string(),
              args: args.to_vec(),
            });
          }
        }
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "IntegerPartitions".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    (1, n.max(1))
  };

  // Parse allowed elements from third arg
  let allowed: Option<Vec<u64>> = if args.len() == 3 {
    match &args[2] {
      Expr::List(elems) => {
        let mut vals = Vec::new();
        for e in elems {
          match e {
            e if expr_to_i128(e).is_some_and(|v| v > 0) => {
              vals.push(expr_to_i128(e).unwrap() as u64)
            }
            _ => {
              return Ok(Expr::FunctionCall {
                name: "IntegerPartitions".to_string(),
                args: args.to_vec(),
              });
            }
          }
        }
        vals.sort_unstable();
        vals.dedup();
        vals.reverse(); // descending for generation
        Some(vals)
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "IntegerPartitions".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    None
  };

  // Special case: n == 0
  // The only partition of 0 is the empty partition {}, which has 0 parts
  if n == 0 {
    if min_len == 0 || (args.len() < 2) {
      return Ok(Expr::List(vec![Expr::List(vec![])]));
    } else {
      return Ok(Expr::List(vec![]));
    }
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
      );
    }
    None => {
      generate_partitions(n, n, max_len, min_len, &mut current, &mut result);
    }
  }

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
pub fn generate_partitions(
  remaining: u64,
  max_part: u64,
  max_len: u64,
  min_len: u64,
  current: &mut Vec<u64>,
  result: &mut Vec<Vec<u64>>,
) {
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
    );
    current.pop();
  }
}

/// Generate partitions using only elements from `elems` (sorted descending).
pub fn generate_partitions_restricted(
  remaining: u64,
  max_len: u64,
  min_len: u64,
  elems: &[u64],
  start_idx: usize,
  current: &mut Vec<u64>,
  result: &mut Vec<Vec<u64>>,
) {
  if remaining == 0 {
    if current.len() as u64 >= min_len {
      result.push(current.clone());
    }
    return;
  }
  if current.len() as u64 >= max_len {
    return;
  }
  for i in start_idx..elems.len() {
    let part = elems[i];
    if part > remaining {
      continue;
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
    );
    current.pop();
  }
}

pub fn divisors_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Divisors expects exactly 1 argument".into(),
    ));
  }

  let n = match expr_to_i128(&args[0]) {
    Some(0) => {
      return Err(InterpreterError::EvaluationError(
        "Divisors: argument cannot be zero".into(),
      ));
    }
    Some(n) => n.unsigned_abs(),
    None => {
      return Ok(Expr::FunctionCall {
        name: "Divisors".to_string(),
        args: args.to_vec(),
      });
    }
  };

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

  divs.sort();
  Ok(Expr::List(
    divs.into_iter().map(|d| Expr::Integer(d as i128)).collect(),
  ))
}

/// DivisorSigma[k, n] - Returns the sum of the k-th powers of divisors of n
pub fn divisor_sigma_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "DivisorSigma expects exactly 2 arguments".into(),
    ));
  }

  let k = match expr_to_i128(&args[0]) {
    Some(k) if k >= 0 => k as u32,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "DivisorSigma: first argument must be a non-negative integer".into(),
      ));
    }
  };

  let n = match expr_to_i128(&args[1]) {
    Some(0) => {
      return Err(InterpreterError::EvaluationError(
        "DivisorSigma: second argument cannot be zero".into(),
      ));
    }
    Some(n) => n.unsigned_abs(),
    None => {
      return Ok(Expr::FunctionCall {
        name: "DivisorSigma".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let sqrt_n = (n as f64).sqrt() as u128;
  let mut sum: u128 = 0;

  for i in 1..=sqrt_n {
    if n % i == 0 {
      sum += i.pow(k);
      if i != n / i {
        sum += (n / i).pow(k);
      }
    }
  }

  Ok(Expr::Integer(sum as i128))
}

/// MoebiusMu[n] - Returns the Möbius function value
pub fn moebius_mu_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "MoebiusMu expects exactly 1 argument".into(),
    ));
  }

  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 1 => n as u128,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "MoebiusMu: argument must be a positive integer".into(),
      ));
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

  let n = match expr_to_i128(&args[0]) {
    Some(0) => return Ok(Expr::Integer(0)),
    Some(n) if n >= 1 => n as u128,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "EulerPhi: argument must be a non-negative integer".into(),
      ));
    }
  };

  let mut num = n;
  let mut result = n;

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

  Ok(Expr::Integer(result as i128))
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
      return Ok(Expr::FunctionCall {
        name: "JacobiSymbol".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let m = match expr_to_i128(&args[1]) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: "JacobiSymbol".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // m must be a positive odd integer
  if m <= 0 || m % 2 == 0 {
    return Err(InterpreterError::EvaluationError(
      "JacobiSymbol: second argument must be a positive odd integer".into(),
    ));
  }

  Ok(Expr::Integer(jacobi_symbol(n, m)))
}

/// Compute the Jacobi symbol (a/n) using the standard algorithm
pub fn jacobi_symbol(mut a: i128, mut n: i128) -> i128 {
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

/// CoprimeQ[a, b, ...] - Tests if integers are pairwise coprime
pub fn coprime_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "CoprimeQ expects at least 2 arguments".into(),
    ));
  }

  // Extract all integer values
  let nums: Vec<u128> = args
    .iter()
    .filter_map(|a| expr_to_i128(a).map(|n| n.unsigned_abs()))
    .collect();

  if nums.len() != args.len() {
    return Ok(Expr::FunctionCall {
      name: "CoprimeQ".to_string(),
      args: args.to_vec(),
    });
  }

  // Check all pairs are coprime
  for i in 0..nums.len() {
    for j in (i + 1)..nums.len() {
      let (mut a, mut b) = (nums[i].max(1), nums[j].max(1));
      while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
      }
      if a != 1 {
        return Ok(Expr::Identifier("False".to_string()));
      }
    }
  }

  Ok(Expr::Identifier("True".to_string()))
}

/// Binomial[n, k] - Binomial coefficient
pub fn binomial_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Binomial expects exactly 2 arguments".into(),
    ));
  }
  match (expr_to_i128(&args[0]), expr_to_i128(&args[1])) {
    (Some(n), Some(k)) => Ok(Expr::Integer(binomial_coeff(n, k))),
    _ => {
      // Try Real evaluation using Gamma function
      let n_f64 = match &args[0] {
        Expr::Real(f) => Some(*f),
        Expr::Integer(n) => Some(*n as f64),
        _ => None,
      };
      let k_f64 = match &args[1] {
        Expr::Real(f) => Some(*f),
        Expr::Integer(n) => Some(*n as f64),
        _ => None,
      };
      if let (Some(n), Some(k)) = (n_f64, k_f64) {
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
        Ok(Expr::FunctionCall {
          name: "Binomial".to_string(),
          args: args.to_vec(),
        })
      }
    }
  }
}

/// Compute binomial coefficient for arbitrary integers (generalized)
pub fn binomial_coeff(n: i128, k: i128) -> i128 {
  if k < 0 {
    return 0;
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
pub fn multinomial_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Integer(1));
  }
  let mut ints = Vec::new();
  for arg in args {
    match expr_to_i128(arg) {
      Some(n) if n >= 0 => ints.push(n),
      Some(_) => {
        return Err(InterpreterError::EvaluationError(
          "Multinomial: arguments must be non-negative integers".into(),
        ));
      }
      None => {
        return Ok(Expr::FunctionCall {
          name: "Multinomial".to_string(),
          args: args.to_vec(),
        });
      }
    }
  }
  // Compute using iterated binomial coefficients: Multinomial[a,b,c] = C(a+b+c, a) * C(b+c, b)
  let mut total: i128 = 0;
  let mut result: i128 = 1;
  for &ni in &ints {
    total += ni;
    result *= binomial_coeff(total, ni);
  }
  Ok(Expr::Integer(result))
}

// ─── PowerMod ──────────────────────────────────────────────────────

/// PowerMod[a, b, m] - Modular exponentiation: a^b mod m
pub fn power_mod_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "PowerMod expects exactly 3 arguments".into(),
    ));
  }
  match (
    expr_to_bigint(&args[0]),
    expr_to_bigint(&args[1]),
    expr_to_bigint(&args[2]),
  ) {
    (Some(base), Some(exp), Some(modulus)) => {
      use num_traits::Zero;
      if modulus.is_zero() {
        eprintln!();
        eprintln!(
          "PowerMod::divz: The argument 0 in PowerMod[{}, {}, 0] should be nonzero.",
          crate::syntax::expr_to_string(&args[0]),
          crate::syntax::expr_to_string(&args[1])
        );
        return Ok(Expr::FunctionCall {
          name: "PowerMod".to_string(),
          args: args.to_vec(),
        });
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
              eprintln!();
              eprintln!(
                "PowerMod::ninv: {} is not invertible modulo {}.",
                crate::syntax::expr_to_string(&args[0]),
                crate::syntax::expr_to_string(&args[2])
              );
              Ok(Expr::FunctionCall {
                name: "PowerMod".to_string(),
                args: args.to_vec(),
              })
            }
          }
          _ => Ok(Expr::FunctionCall {
            name: "PowerMod".to_string(),
            args: args.to_vec(),
          }),
        }
      } else {
        let result = base.modpow(&exp, &modulus);
        // Ensure non-negative result
        let result = ((result % &modulus) + &modulus) % &modulus;
        Ok(bigint_to_expr(result))
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "PowerMod".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Binary exponentiation: base^exp mod modulus (all unsigned)
/// Uses BigUint for intermediate multiplication to avoid u128 overflow.
pub fn mod_pow_unsigned(base: u128, mut exp: u128, modulus: u128) -> u128 {
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
pub fn mod_inverse(a: i128, m: i128) -> Option<i128> {
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
      let n_usize = n as usize;
      let mut count: i128 = 0;
      for i in 2..=n_usize {
        if crate::is_prime(i) {
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
      let n = f.floor() as usize;
      let mut count: i128 = 0;
      for i in 2..=n {
        if crate::is_prime(i) {
          count += 1;
        }
      }
      Ok(Expr::Integer(count))
    }
    _ => Ok(Expr::FunctionCall {
      name: "PrimePi".to_string(),
      args: args.to_vec(),
    }),
  }
}

// ─── BigInt Primality (Miller-Rabin) ──────────────────────────────

/// Miller-Rabin primality test for BigInt values.
/// Uses deterministic witnesses for small numbers and a set of strong
/// witnesses that provides correct results for all numbers < 3.317e24,
/// plus additional witnesses for larger numbers.
pub fn is_prime_bigint(n: &num_bigint::BigInt) -> bool {
  use num_bigint::BigInt;
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
        return Ok(Expr::FunctionCall {
          name: "NextPrime".to_string(),
          args: args.to_vec(),
        });
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
    return Ok(Expr::FunctionCall {
      name: "NextPrime".to_string(),
      args: args.to_vec(),
    });
  }

  let n = match &args[0] {
    Expr::Integer(n) => *n,
    Expr::Real(f) => f.floor() as i128,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "NextPrime".to_string(),
        args: args.to_vec(),
      });
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
    Ok(Expr::FunctionCall {
      name: "NextPrime".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Find the smallest prime > n (including negative primes like -2, -3, -5, ...).
pub fn next_prime_after(n: i128) -> i128 {
  // For n >= 1: search upward from n+1
  if n >= 1 {
    let mut candidate = n + 1;
    while !crate::is_prime(candidate as usize) {
      candidate += 1;
    }
    return candidate;
  }
  // For n < -2: check negative primes between n and -2 (exclusive of n, inclusive of -2)
  // A negative prime is -p where p is a positive prime.
  // Search from n+1 upward: for each candidate c, check if |c| is prime.
  if n < -2 {
    for c in (n + 1)..=-2 {
      if c < 0 && crate::is_prime((-c) as usize) {
        return c;
      }
    }
  }
  // No negative prime found > n, or n is -2, -1, or 0: smallest positive prime is 2
  2
}

/// Find the largest prime < n (including negative primes).
/// The prime sequence is: ..., -7, -5, -3, -2, 2, 3, 5, 7, 11, ...
pub fn prev_prime_before(n: i128) -> i128 {
  // For n > 2: search downward among positive primes
  if n > 2 {
    let mut candidate = n - 1;
    while candidate >= 2 {
      if crate::is_prime(candidate as usize) {
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
    if crate::is_prime((-candidate) as usize) {
      return candidate;
    }
    candidate -= 1;
  }
}

/// Find the smallest prime > n for BigInt values.
pub fn next_prime_after_bigint(n: &num_bigint::BigInt) -> num_bigint::BigInt {
  use num_bigint::BigInt;
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
  let a = match expr_to_bigint(&args[0]) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: "ModularInverse".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let m = match expr_to_bigint(&args[1]) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: "ModularInverse".to_string(),
        args: args.to_vec(),
      });
    }
  };

  use num_traits::{One, Zero};

  if m.is_zero() {
    return Ok(Expr::FunctionCall {
      name: "ModularInverse".to_string(),
      args: args.to_vec(),
    });
  }

  // Extended Euclidean algorithm
  let m_abs = if m < BigInt::zero() {
    -m.clone()
  } else {
    m.clone()
  };
  let (gcd, x, _) = extended_gcd(&a, &m_abs);
  if !gcd.is_one() && gcd != -BigInt::one() {
    // Not coprime, no inverse exists - return unevaluated
    return Ok(Expr::FunctionCall {
      name: "ModularInverse".to_string(),
      args: args.to_vec(),
    });
  }

  // Normalize result to be in range [0, |m|-1]
  let result = ((x % &m_abs) + &m_abs) % &m_abs;
  Ok(bigint_to_expr(result))
}

/// Extended GCD: returns (gcd, x, y) such that a*x + b*y = gcd
pub fn extended_gcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
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
    None => Ok(Expr::FunctionCall {
      name: "BitLength".to_string(),
      args: args.to_vec(),
    }),
  }
}

// ─── Bitwise operations ──────────────────────────────────────────

/// BitAnd[n1, n2, ...] - bitwise AND
pub fn bit_and_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
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
        return Ok(Expr::FunctionCall {
          name: "BitAnd".to_string(),
          args: args.to_vec(),
        });
      }
    }
  }
  match result {
    Some(n) => Ok(bigint_to_expr(n)),
    None => Ok(Expr::FunctionCall {
      name: "BitAnd".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// BitOr[n1, n2, ...] - bitwise OR
pub fn bit_or_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
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
        return Ok(Expr::FunctionCall {
          name: "BitOr".to_string(),
          args: args.to_vec(),
        });
      }
    }
  }
  match result {
    Some(n) => Ok(bigint_to_expr(n)),
    None => Ok(Expr::FunctionCall {
      name: "BitOr".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// BitXor[n1, n2, ...] - bitwise XOR
pub fn bit_xor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
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
        return Ok(Expr::FunctionCall {
          name: "BitXor".to_string(),
          args: args.to_vec(),
        });
      }
    }
  }
  match result {
    Some(n) => Ok(bigint_to_expr(n)),
    None => Ok(Expr::FunctionCall {
      name: "BitXor".to_string(),
      args: args.to_vec(),
    }),
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
    None => Ok(Expr::FunctionCall {
      name: "BitNot".to_string(),
      args: args.to_vec(),
    }),
  }
}

// ─── IntegerExponent ──────────────────────────────────────────────

/// FrobeniusNumber[{a1, a2, ...}] - Largest integer that cannot be represented
/// as a non-negative integer linear combination of the given positive integers.
/// Returns Infinity if the GCD is not 1, -1 if 1 is in the set.
pub fn frobenius_number_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "FrobeniusNumber".to_string(),
      args: args.to_vec(),
    });
  }

  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "FrobeniusNumber".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if items.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "FrobeniusNumber".to_string(),
      args: args.to_vec(),
    });
  }

  // Extract positive integers
  let mut nums: Vec<i128> = Vec::new();
  for item in items {
    match item {
      e if expr_to_i128(e).is_some_and(|n| n > 0) => {
        nums.push(expr_to_i128(e).unwrap())
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "FrobeniusNumber".to_string(),
          args: args.to_vec(),
        });
      }
    }
  }

  // If 1 is in the set, every non-negative integer is representable
  if nums.contains(&1) {
    return Ok(Expr::Integer(-1));
  }

  // Compute GCD of all elements
  fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
  }

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
    None => Ok(Expr::FunctionCall {
      name: "PartitionsP".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Compute p(n) using dynamic programming
pub fn partitions_p(n: usize) -> BigInt {
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
    None => Ok(Expr::FunctionCall {
      name: "PartitionsQ".to_string(),
      args: args.to_vec(),
    }),
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

  // AGM[0, b] = 0 and AGM[a, 0] = 0
  let is_zero = |e: &Expr| {
    matches!(e, Expr::Integer(0)) || matches!(e, Expr::Real(x) if *x == 0.0)
  };
  if is_zero(&args[0]) || is_zero(&args[1]) {
    return Ok(Expr::Integer(0));
  }

  // AGM[a, a] = a (for integer/real values)
  if let (Some(a), Some(b)) = (expr_to_f64(&args[0]), expr_to_f64(&args[1]))
    && a == b
  {
    return Ok(args[0].clone());
  }

  // Numeric evaluation when both are numeric and at least one is Real
  if let (Some(a), Some(b)) = (expr_to_f64(&args[0]), expr_to_f64(&args[1])) {
    let is_numeric_eval =
      matches!(&args[0], Expr::Real(_)) || matches!(&args[1], Expr::Real(_));
    if is_numeric_eval {
      return Ok(Expr::Real(agm(a, b)));
    }
  }

  // Stay unevaluated for symbolic/exact inputs
  Ok(Expr::FunctionCall {
    name: "ArithmeticGeometricMean".to_string(),
    args: args.to_vec(),
  })
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
