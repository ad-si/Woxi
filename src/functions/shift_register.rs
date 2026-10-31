//! `ShiftRegisterSequence` — the output of a linear feedback shift register.
//!
//! For a characteristic polynomial `c_0 + c_1 x + … + c_n x^n` over the
//! integers mod `m` the register starts at `0, …, 0, 1` (`n - 1` zeros) and
//! steps `s[k + n] = -(c_0 s[k] + … + c_{n-1} s[k + n - 1]) / c_n`; a
//! primitive polynomial gives the full period `m^n - 1`, which is the
//! default length. `ShiftRegisterSequence[n]` uses the binary primitive
//! polynomial of degree `n` with the fewest terms, the one whose exponents
//! come first lexicographically — wolframscript's choice for every degree
//! checked (2–18).

use super::*;

/// The sequence for coefficients `c` (constant term first) mod `m`.
fn register_sequence(c: &[i128], m: i128, len: usize) -> Option<Vec<i128>> {
  let n = c.len() - 1;
  let lead = c[n].rem_euclid(m);
  // The inverse of the leading coefficient mod m.
  let inv = (1..m).find(|k| (k * lead).rem_euclid(m) == 1)?;
  let mut s: Vec<i128> = vec![0; n.saturating_sub(1)];
  s.push(1);
  while s.len() < len {
    let k = s.len() - n;
    let sum: i128 = (0..n).map(|i| c[i] * s[k + i]).sum();
    s.push((-sum * inv).rem_euclid(m));
  }
  s.truncate(len);
  Some(s)
}

/// `a * b mod p` over GF(2), polynomials as bit masks, `p` of degree `n`.
fn gf2_mulmod(mut a: u64, mut b: u64, p: u64, n: u32) -> u64 {
  let mut r = 0u64;
  while b != 0 {
    if b & 1 == 1 {
      r ^= a;
    }
    b >>= 1;
    a <<= 1;
    if a >> n & 1 == 1 {
      a ^= p;
    }
  }
  r
}

fn gf2_x_pow(e: u64, p: u64, n: u32) -> u64 {
  let (mut result, mut base, mut e) = (1u64, 2u64, e);
  while e > 0 {
    if e & 1 == 1 {
      result = gf2_mulmod(result, base, p, n);
    }
    base = gf2_mulmod(base, base, p, n);
    e >>= 1;
  }
  result
}

/// Whether `p` (degree `n`, constant term 1) is primitive over GF(2): `x`
/// has order exactly `2^n - 1` modulo `p`.
fn gf2_primitive(p: u64, n: u32, prime_factors: &[u64]) -> bool {
  let order = (1u64 << n) - 1;
  gf2_x_pow(order, p, n) == 1
    && prime_factors
      .iter()
      .all(|q| gf2_x_pow(order / q, p, n) != 1)
}

fn prime_factors(mut v: u64) -> Vec<u64> {
  let mut out = Vec::new();
  let mut f = 2u64;
  while f * f <= v {
    if v.is_multiple_of(f) {
      out.push(f);
      while v.is_multiple_of(f) {
        v /= f;
      }
    }
    f += 1;
  }
  if v > 1 {
    out.push(v);
  }
  out
}

/// The default binary primitive polynomial of degree `n`, as coefficients.
fn default_polynomial(n: u32) -> Option<Vec<i128>> {
  if !(2..=32).contains(&n) {
    return None;
  }
  let factors = prime_factors((1u64 << n) - 1);
  // Middle exponents in lexicographic order, fewest first.
  fn search(
    start: u32,
    left: u32,
    n: u32,
    chosen: &mut Vec<u32>,
    found: &mut dyn FnMut(&[u32]) -> bool,
  ) -> bool {
    if left == 0 {
      return found(chosen);
    }
    for e in start..n {
      chosen.push(e);
      if search(e + 1, left - 1, n, chosen, found) {
        return true;
      }
      chosen.pop();
    }
    false
  }
  for middle in (1..n).step_by(2) {
    let mut result: Option<Vec<u32>> = None;
    let mut test = |exps: &[u32]| {
      let p = exps
        .iter()
        .fold(1u64 | (1u64 << n), |acc, e| acc | (1 << e));
      if gf2_primitive(p, n, &factors) {
        result = Some(exps.to_vec());
        true
      } else {
        false
      }
    };
    search(1, middle, n, &mut Vec::new(), &mut test);
    if let Some(exps) = result {
      let mut c = vec![0i128; n as usize + 1];
      c[0] = 1;
      c[n as usize] = 1;
      for e in exps {
        c[e as usize] = 1;
      }
      return Some(c);
    }
  }
  None
}

/// The coefficients of a univariate polynomial with integer coefficients.
fn polynomial_coefficients(poly: &Expr) -> Option<Vec<i128>> {
  let Ok(Expr::List(ref vars)) = crate::evaluator::evaluate_function_call_ast(
    "Variables",
    std::slice::from_ref(poly),
  ) else {
    return None;
  };
  let [var] = vars.as_slice() else {
    return None;
  };
  let Ok(Expr::List(ref coeffs)) = crate::evaluator::evaluate_function_call_ast(
    "CoefficientList",
    &[poly.clone(), var.clone()],
  ) else {
    return None;
  };
  let c: Vec<i128> = coeffs
    .iter()
    .map(|c| match c {
      Expr::Integer(k) => Some(*k),
      _ => None,
    })
    .collect::<Option<_>>()?;
  (c.len() >= 2).then_some(c)
}

pub fn shift_register_sequence_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated =
    || Ok(crate::helpers::unevaluated("ShiftRegisterSequence", args));
  let mut modulus: i128 = 2;
  let mut positional: Vec<&Expr> = Vec::new();
  for a in args {
    match a {
      Expr::Rule {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Modulus") => {
        match replacement.as_ref() {
          Expr::Integer(m) if *m >= 2 => modulus = *m,
          _ => return unevaluated(),
        }
      }
      other => positional.push(other),
    }
  }
  let (spec, len) = match positional.as_slice() {
    [spec] => (*spec, None),
    [spec, Expr::Integer(len)] if *len >= 0 => (*spec, Some(*len as usize)),
    _ => return unevaluated(),
  };
  let coeffs = match spec {
    Expr::Integer(n) if *n >= 2 && modulus == 2 => {
      match default_polynomial(*n as u32) {
        Some(c) => c,
        None => return unevaluated(),
      }
    }
    Expr::Integer(_) | Expr::List(_) => return unevaluated(),
    poly => match polynomial_coefficients(poly) {
      Some(c) => c,
      None => return unevaluated(),
    },
  };
  let degree = (coeffs.len() - 1) as u32;
  let len = match len {
    Some(len) => len,
    None => match modulus.checked_pow(degree) {
      Some(p) if p <= 1 << 24 => (p - 1) as usize,
      _ => return unevaluated(),
    },
  };
  let Some(seq) = register_sequence(&coeffs, modulus, len) else {
    return unevaluated();
  };
  Ok(Expr::List(seq.into_iter().map(Expr::Integer).collect()))
}
