//! DirichletCharacter[k, j, n] — the j-th Dirichlet character modulo k
//! evaluated at n, following Wolfram's indexing convention (decoded
//! from wolframscript tables for k = 3, 5, 8, 9, 12, 15, 16, 24):
//!
//! (Z/k)* is factored over the prime powers of k in ascending prime
//! order; an odd p^e contributes one cyclic factor on its smallest
//! primitive root, 2^2 contributes <-1>, and 2^e (e >= 3) contributes
//! <-1> then <5>. The index j-1 is expanded in mixed radix over the
//! factor orders with the LAST factor least significant, giving the
//! character's exponent on each factor generator. Values are roots of
//! unity e^(2 pi i r), printed exactly as wolframscript does
//! (1, -1, I, -I, or E^((a*I)/b*Pi) forms on the principal branch).

use crate::InterpreterError;
use crate::syntax::Expr;

pub fn dirichlet_character_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "DirichletCharacter".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 3 {
    return Ok(unevaluated(args));
  }
  let k = match &args[0] {
    Expr::Integer(k) if *k >= 1 => *k,
    _ => return Ok(unevaluated(args)),
  };
  let j = match &args[1] {
    Expr::Integer(j) => *j,
    _ => return Ok(unevaluated(args)),
  };
  if j <= 0 {
    crate::emit_message(&format!(
      "DirichletCharacter::intp: Positive integer expected at position 2 in {}.",
      crate::syntax::expr_to_string(&unevaluated(args))
    ));
    return Ok(unevaluated(args));
  }
  let phi = euler_phi(k);
  if j > phi {
    crate::emit_message(&format!(
      "DirichletCharacter::invl: Argument {} at position 2 in {} should be a positive integer less than or equal to {}.",
      j,
      crate::syntax::expr_to_string(&unevaluated(args)),
      phi
    ));
    return Ok(unevaluated(args));
  }
  let n = match &args[2] {
    Expr::Integer(n) => n.rem_euclid(k),
    _ => return Ok(unevaluated(args)),
  };

  if gcd(n, k) != 1 {
    // k = 1 has gcd(n, 1) = 1 for every n, so this is unreachable there
    return Ok(Expr::Integer(0));
  }

  // Factor generators in mixed-radix order (most significant first)
  let factors = character_factors(k);
  // Exponents of the j-th character: j-1 in mixed radix, last factor
  // least significant
  let mut exps = vec![0i128; factors.len()];
  let mut rest = j - 1;
  for (i, f) in factors.iter().enumerate().rev() {
    exps[i] = rest % f.order;
    rest /= f.order;
  }

  // chi(n) = product of per-factor roots of unity e^(2 pi i e_i t_i / o_i).
  // Wolfram folds the fourth roots (1, -1, I, -I) into a numeric
  // coefficient and keeps one combined exponential symbolic, so track
  // the two parts separately to reproduce its print forms.
  let mut quarter = 0i128; // coefficient exponent: i^quarter
  let (mut num, mut den) = (0i128, 1i128); // exponential part, mod 1
  for (f, e) in factors.iter().zip(&exps) {
    let t = f.dlog(n);
    let (mut p, mut q) = (e * t, f.order);
    let g = gcd(p, q);
    p = (p / g).rem_euclid(q / g);
    q /= g;
    if q == 1 || q == 2 || q == 4 {
      quarter += p * (4 / q);
    } else {
      num = num * q + p * den;
      den *= q;
      let g = gcd(num, den);
      num /= g;
      den /= g;
    }
  }
  num = num.rem_euclid(den);
  let g = gcd(num, den);
  num /= g;
  den /= g;
  // A combined exponential that lands on a fourth root folds into the
  // coefficient too
  if den == 1 || den == 2 || den == 4 {
    quarter += num * (4 / den);
    num = 0;
    den = 1;
  }
  quarter = quarter.rem_euclid(4);

  Ok(assemble_value(quarter, num, den))
}

/// One cyclic factor of (Z/k)*: dlog maps n (mod k, coprime) to the
/// exponent of this factor's generator in n's decomposition.
struct Factor {
  modulus: i128, // the prime power
  generator: i128,
  order: i128,
  two_part_sign: bool, // the <-1> factor of 2^e (e >= 3)
}

impl Factor {
  fn dlog(&self, n: i128) -> i128 {
    let m = n.rem_euclid(self.modulus);
    if self.two_part_sign {
      // n = (-1)^s * 5^t mod 2^e: s = 0 iff m is a power of 5
      let mut p = 1i128;
      let order5 = self.modulus / 4;
      for _ in 0..order5 {
        if p == m {
          return 0;
        }
        p = p * 5 % self.modulus;
      }
      return 1;
    }
    // The <5> factor of 2^e (e >= 3) sees elements only up to sign:
    // n = (-1)^s * 5^t, so match against both 5^t and -5^t. (The mod-4
    // factor <3> and odd prime powers must match exactly.)
    let up_to_sign = self.modulus % 2 == 0 && self.generator == 5;
    let mut p = 1i128;
    for t in 0..self.order {
      if p == m || (up_to_sign && self.modulus - p == m) {
        return t;
      }
      p = p * self.generator % self.modulus;
    }
    unreachable!("dlog: generator does not generate the group")
  }
}

/// Mixed-radix factor list for (Z/k)*, most significant first:
/// prime powers in ascending prime order; 2^e (e >= 3) contributes
/// <-1> then <5>.
fn character_factors(k: i128) -> Vec<Factor> {
  let mut factors = Vec::new();
  let mut rest = k;
  let mut two_e = 0;
  while rest % 2 == 0 {
    rest /= 2;
    two_e += 1;
  }
  let two_modulus = k / rest; // 2^two_e
  if two_e == 2 {
    factors.push(Factor {
      modulus: 4,
      generator: 3, // -1 mod 4
      order: 2,
      two_part_sign: false,
    });
  } else if two_e >= 3 {
    factors.push(Factor {
      modulus: two_modulus,
      generator: two_modulus - 1, // -1
      order: 2,
      two_part_sign: true,
    });
    factors.push(Factor {
      modulus: two_modulus,
      generator: 5,
      order: two_modulus / 4,
      two_part_sign: false,
    });
  }
  let mut p = 3i128;
  while p * p <= rest || rest > 1 {
    if rest % p == 0 {
      let mut pe = 1i128;
      while rest % p == 0 {
        rest /= p;
        pe *= p;
      }
      let order = pe / p * (p - 1); // phi(p^e)
      factors.push(Factor {
        modulus: pe,
        generator: primitive_root(pe, order),
        order,
        two_part_sign: false,
      });
    }
    p += 2;
    if p * p > rest && rest > 1 {
      let pe_p = rest;
      // rest is now a single prime
      let order = pe_p - 1;
      factors.push(Factor {
        modulus: pe_p,
        generator: primitive_root(pe_p, order),
        order,
        two_part_sign: false,
      });
      rest = 1;
    }
  }
  factors
}

/// Smallest primitive root modulo the odd prime power pe.
fn primitive_root(pe: i128, order: i128) -> i128 {
  'g: for g in 2..pe {
    if gcd(g, pe) != 1 {
      continue;
    }
    // g is a primitive root iff g^(order/q) != 1 for every prime q | order
    let mut o = order;
    let mut q = 2i128;
    let mut prime_divisors = Vec::new();
    while q * q <= o {
      if o % q == 0 {
        prime_divisors.push(q);
        while o % q == 0 {
          o /= q;
        }
      }
      q += 1;
    }
    if o > 1 {
      prime_divisors.push(o);
    }
    for q in prime_divisors {
      if pow_mod(g, order / q, pe) == 1 {
        continue 'g;
      }
    }
    return g;
  }
  unreachable!("no primitive root found")
}

fn pow_mod(mut b: i128, mut e: i128, m: i128) -> i128 {
  let mut r = 1i128;
  b %= m;
  while e > 0 {
    if e & 1 == 1 {
      r = r * b % m;
    }
    b = b * b % m;
    e >>= 1;
  }
  r
}

fn gcd(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    (a, b) = (b, a % b);
  }
  a.max(1)
}

fn euler_phi(k: i128) -> i128 {
  let mut result = k;
  let mut n = k;
  let mut p = 2i128;
  while p * p <= n {
    if n % p == 0 {
      while n % p == 0 {
        n /= p;
      }
      result -= result / p;
    }
    p += 1;
  }
  if n > 1 {
    result -= result / n;
  }
  result
}

/// i^quarter * e^(2 pi i num/den) in wolframscript's exact print
/// forms. The exponential exponent is reduced to the principal branch
/// (-pi, pi]; an I or -I coefficient over a negative exponent prints
/// as a quotient (Times with a negative power formats as division).
fn assemble_value(quarter: i128, num: i128, den: i128) -> Expr {
  if num == 0 {
    return match quarter {
      0 => Expr::Integer(1),
      1 => Expr::Identifier("I".to_string()),
      2 => Expr::Integer(-1),
      _ => Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(Expr::Identifier("I".to_string())),
      },
    };
  }
  // Principal branch: the printed multiple of Pi is m = a/b = 2*num/den
  // reduced into (-1, 1]
  let (mut a, mut b) = (2 * num, den);
  let g = gcd(a, b);
  a /= g;
  b /= g;
  if a > b {
    a -= 2 * b;
    let g = gcd(a, b);
    a /= g;
    b /= g;
  }
  let exp_form = |a: i128, b: i128| -> String {
    let inner = if a == 1 {
      format!("I/{b}")
    } else if a == -1 {
      format!("(-1/{b}*I)")
    } else {
      format!("({a}*I)/{b}")
    };
    format!("E^({inner}*Pi)")
  };
  Expr::Raw(match quarter {
    0 => exp_form(a, b),
    2 => format!("-{}", exp_form(a, b)),
    q => {
      let c = if q == 1 { "I" } else { "-I" };
      if a > 0 {
        format!("{c}*{}", exp_form(a, b))
      } else {
        format!("{c}/{}", exp_form(-a, b))
      }
    }
  })
}

/// Total rotation r in [0, 1) (as a reduced fraction) with
/// chi_j(a) = e^(2 pi i r), for a coprime to k.
fn total_rotation(factors: &[Factor], exps: &[i128], a: i128) -> (i128, i128) {
  let (mut num, mut den) = (0i128, 1i128);
  for (f, e) in factors.iter().zip(exps) {
    let t = f.dlog(a);
    let (p, q) = (e * t, f.order);
    num = num * q + p * den;
    den *= q;
    let g = gcd(num, den);
    num /= g;
    den /= g;
  }
  num = num.rem_euclid(den);
  let g = gcd(num, den);
  (num / g, den / g)
}

/// DirichletL[k, j, s] - Dirichlet L-function of the j-th character
/// modulo k, in the cases wolframscript evaluates exactly:
/// - k == 1: Zeta[s];
/// - non-principal characters with values in {1, -1, I, -I} at integer
///   s <= 0: exact (complex-)rational values via generalized Bernoulli
///   numbers, L(1-n, chi) = -B_{n,chi}/n;
/// - odd real characters at s == 1 for k <= 4: the classic Pi forms
///   via the cotangent sum (higher conductors use Wolfram-internal
///   radical/log canonicalizations we do not reproduce).
/// Principal characters (k > 1), s >= 2, non-integer s, and characters
/// with higher-order values stay unevaluated.
pub fn dirichlet_l_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use num_bigint::BigInt;
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "DirichletL".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 3 {
    return Ok(unevaluated(args));
  }
  let k = match &args[0] {
    Expr::Integer(k) if *k >= 1 => *k,
    _ => return Ok(unevaluated(args)),
  };
  let j = match &args[1] {
    Expr::Integer(j) => *j,
    _ => return Ok(unevaluated(args)),
  };
  if j <= 0 {
    crate::emit_message(&format!(
      "DirichletL::intp: Positive integer expected at position 2 in {}.",
      crate::syntax::expr_to_string(&unevaluated(args))
    ));
    return Ok(unevaluated(args));
  }
  let phi = euler_phi(k);
  if j > phi {
    crate::emit_message(&format!(
      "DirichletL::invl: Argument {} at position 2 in {} should be a positive integer less than or equal to {}.",
      j,
      crate::syntax::expr_to_string(&unevaluated(args)),
      phi
    ));
    return Ok(unevaluated(args));
  }
  let s = match &args[2] {
    Expr::Integer(s) => *s,
    _ => return Ok(unevaluated(args)),
  };

  if k == 1 {
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Zeta".to_string(),
      args: vec![Expr::Integer(s)].into(),
    });
  }
  if j == 1 {
    // Principal characters never evaluate (the s = 1 series diverges
    // and wolframscript keeps the rest symbolic as well)
    return Ok(unevaluated(args));
  }

  let factors = character_factors(k);
  let mut exps = vec![0i128; factors.len()];
  let mut rest = j - 1;
  for (i, f) in factors.iter().enumerate().rev() {
    exps[i] = rest % f.order;
    rest /= f.order;
  }
  // chi(a) for a = 1..k as rotations; None where chi vanishes
  let rotations: Vec<Option<(i128, i128)>> = (1..=k)
    .map(|a| {
      if gcd(a, k) == 1 {
        Some(total_rotation(&factors, &exps, a))
      } else {
        None
      }
    })
    .collect();

  if s == 1 {
    // Odd real character: L(1) = (Pi/(2k)) * sum chi(a) Cot[Pi a/k]
    let real = rotations
      .iter()
      .flatten()
      .all(|&(_, den)| den == 1 || den == 2);
    let odd = rotations[(k - 2) as usize] == Some((1, 2)); // chi(k-1) = -1
    if !(real && odd && k <= 4) {
      return Ok(unevaluated(args));
    }
    let mut terms: Vec<Expr> = Vec::new();
    for (a, rot) in rotations.iter().enumerate().take((k - 1) as usize) {
      let Some(&(num, _)) = rot.as_ref() else {
        continue;
      };
      let cot = Expr::FunctionCall {
        name: "Cot".to_string(),
        args: vec![Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(a as i128 + 1), Expr::Integer(k)].into(),
            },
            Expr::Constant("Pi".to_string()),
          ]
          .into(),
        }]
        .into(),
      };
      terms.push(if num == 0 {
        cot
      } else {
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), cot].into(),
        }
      });
    }
    let sum = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    };
    let product = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(1), Expr::Integer(2 * k)].into(),
        },
        Expr::Constant("Pi".to_string()),
        sum,
      ]
      .into(),
    };
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Simplify".to_string(),
      args: vec![product].into(),
    });
  }

  if s > 1 {
    return Ok(unevaluated(args));
  }

  // s <= 0: L(1-n, chi) = -B_{n,chi}/n with n = 1-s, where
  // B_{n,chi} = k^(n-1) * sum_a chi(a) B_n(a/k). Exact only when the
  // character values are fourth roots of unity.
  if !rotations.iter().flatten().all(|&(_, den)| den <= 4) {
    return Ok(unevaluated(args));
  }
  let n = (1 - s) as usize;

  // Fraction helpers over BigInt
  type Frac = (BigInt, BigInt);
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
  let reduce = |num: BigInt, den: BigInt| -> Frac {
    let g = gcd_bigint(&num, &den);
    let (mut n, mut d) = (num / &g, den / g);
    if d < BigInt::from(0) {
      n = -n;
      d = -d;
    }
    (n, d)
  };
  let add = |a: &Frac, b: &Frac| reduce(&a.0 * &b.1 + &b.0 * &a.1, &a.1 * &b.1);
  let mul = |a: &Frac, b: &Frac| reduce(&a.0 * &b.0, &a.1 * &b.1);

  // Bernoulli numbers B_0..B_n (B_1 = -1/2) via the standard recurrence
  let mut bernoulli: Vec<Frac> = Vec::with_capacity(n + 1);
  for m in 0..=n {
    if m == 0 {
      bernoulli.push((BigInt::from(1), BigInt::from(1)));
      continue;
    }
    // B_m = -1/(m+1) * sum_{i<m} C(m+1, i) B_i
    let mut acc: Frac = (BigInt::from(0), BigInt::from(1));
    let mut binom = BigInt::from(1); // C(m+1, 0)
    for (i, b) in bernoulli.iter().enumerate().take(m) {
      let term = mul(&(binom.clone(), BigInt::from(1)), b);
      acc = add(&acc, &term);
      binom =
        binom * BigInt::from((m + 1 - i) as i64) / BigInt::from((i + 1) as i64);
    }
    bernoulli.push(reduce(-acc.0, acc.1 * BigInt::from((m + 1) as i64)));
  }
  // Binomials C(n, i)
  let mut binoms: Vec<BigInt> = Vec::with_capacity(n + 1);
  let mut c = BigInt::from(1);
  for i in 0..=n {
    binoms.push(c.clone());
    if i < n {
      c = c * BigInt::from((n - i) as i64) / BigInt::from((i + 1) as i64);
    }
  }

  // B_{n,chi} = k^(n-1) sum_a chi(a) B_n(a/k), complex (re, im)
  let mut re: Frac = (BigInt::from(0), BigInt::from(1));
  let mut im: Frac = (BigInt::from(0), BigInt::from(1));
  for (idx, rot) in rotations.iter().enumerate() {
    let Some(&(rnum, rden)) = rot.as_ref() else {
      continue;
    };
    let a = idx as i128 + 1;
    // B_n(a/k) = sum_i C(n,i) B_i (a/k)^(n-i)
    let x: Frac = reduce(BigInt::from(a), BigInt::from(k));
    let mut poly: Frac = (BigInt::from(0), BigInt::from(1));
    let mut x_pow: Frac = (BigInt::from(1), BigInt::from(1));
    // accumulate from i = n down to 0 so x_pow tracks x^(n-i)
    for i in (0..=n).rev() {
      let term = mul(
        &mul(&(binoms[i].clone(), BigInt::from(1)), &bernoulli[i]),
        &x_pow,
      );
      poly = add(&poly, &term);
      x_pow = mul(&x_pow, &x);
    }
    // chi(a): rotation num/den with den | 4
    let (cre, cim): (i64, i64) = match (rnum * (4 / rden)).rem_euclid(4) {
      0 => (1, 0),
      1 => (0, 1),
      2 => (-1, 0),
      _ => (0, -1),
    };
    if cre != 0 {
      re = add(&re, &mul(&(BigInt::from(cre), BigInt::from(1)), &poly));
    }
    if cim != 0 {
      im = add(&im, &mul(&(BigInt::from(cim), BigInt::from(1)), &poly));
    }
  }
  // Scale by k^(n-1) and -1/n
  let mut k_pow = (BigInt::from(1), BigInt::from(1));
  if n >= 1 {
    k_pow = (BigInt::from(k).pow((n - 1) as u32), BigInt::from(1));
  }
  let scale = mul(&k_pow, &(BigInt::from(-1), BigInt::from(n as i64)));
  re = mul(&re, &scale);
  im = mul(&im, &scale);

  let re_expr = crate::functions::make_rational_expr(re.0, re.1);
  let im_is_zero = im.0 == BigInt::from(0);
  if im_is_zero {
    return Ok(re_expr);
  }
  let im_expr = crate::functions::make_rational_expr(im.0, im.1);
  crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      re_expr,
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![im_expr, Expr::Identifier("I".to_string())].into(),
      },
    ]
    .into(),
  })
}

// ---------------------------------------------------------------------------
// DirichletConvolve[f, g, n, m] — the Dirichlet convolution
//   Sum over the divisors d of m of f(d) g(m/d).
//
// Matches wolframscript's evaluation strategy (decoded by probing):
// 1. Positive integer m: expand the divisor sum directly.
// 2. Symbolic m, every term pair reducible by the identity table
//    (n^j * n^k, MoebiusMu * 1, MoebiusMu * n, EulerPhi * 1): full closed
//    form via linearity.
// 3. Symbolic m with g == 1: rewrite as DivisorSum[m, f(#) &] (whole f —
//    no linear split on this path).
// 4. Otherwise split linearly (swapping the arguments first when f is a
//    sum and g is not); reducible pairs use the table plus
//    `unknown * 1 -> DivisorSum`, irreducible pairs stay as inert
//    DirichletConvolve terms.

use crate::syntax::BinaryOperator;

#[derive(Clone)]
enum ConvAtom {
  /// var^k for integer k >= 0 (k = 0 is the constant 1, k = 1 is var)
  Power(i128),
  Mu,
  Phi,
  Unknown,
}

/// One additive term of f or g: numeric/var-free coefficient factors,
/// the classified var-dependent atom, and the atom's original expression
/// (used for inert DirichletConvolve output and DivisorSum bodies).
struct ConvTerm {
  coeff: Vec<Expr>,
  atom: ConvAtom,
  atom_expr: Expr,
}

fn flatten_plus(expr: &Expr, out: &mut Vec<Expr>) {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      flatten_plus(left, out);
      flatten_plus(right, out);
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      for a in args.iter() {
        flatten_plus(a, out);
      }
    }
    _ => out.push(expr.clone()),
  }
}

fn flatten_times(expr: &Expr, out: &mut Vec<Expr>) {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      flatten_times(left, out);
      flatten_times(right, out);
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      for a in args.iter() {
        flatten_times(a, out);
      }
    }
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      out.push(Expr::Integer(-1));
      flatten_times(operand, out);
    }
    _ => out.push(expr.clone()),
  }
}

fn classify_term(term: &Expr, var: &str) -> ConvTerm {
  let mut factors = Vec::new();
  flatten_times(term, &mut factors);
  let (coeff, var_factors): (Vec<Expr>, Vec<Expr>) = factors
    .into_iter()
    .partition(|f| crate::functions::calculus_ast::is_constant_wrt(f, var));
  let atom = if var_factors.is_empty() {
    ConvAtom::Power(0)
  } else if var_factors.len() == 1 {
    match &var_factors[0] {
      Expr::Identifier(s) if s == var => ConvAtom::Power(1),
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } if matches!(left.as_ref(), Expr::Identifier(s) if s == var) => {
        match right.as_ref() {
          Expr::Integer(k) if *k >= 0 => ConvAtom::Power(*k),
          _ => ConvAtom::Unknown,
        }
      }
      Expr::FunctionCall { name, args }
        if args.len() == 1
          && matches!(&args[0], Expr::Identifier(s) if s == var) =>
      {
        match name.as_str() {
          "MoebiusMu" => ConvAtom::Mu,
          "EulerPhi" => ConvAtom::Phi,
          _ => ConvAtom::Unknown,
        }
      }
      _ => ConvAtom::Unknown,
    }
  } else {
    ConvAtom::Unknown
  };
  let atom_expr = if var_factors.is_empty() {
    Expr::Integer(1)
  } else if var_factors.len() == 1 {
    var_factors[0].clone()
  } else {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: var_factors.into(),
    }
  };
  ConvTerm {
    coeff,
    atom,
    atom_expr,
  }
}

fn divisor_sigma_expr(k: i128, m: &Expr) -> Expr {
  Expr::FunctionCall {
    name: "DivisorSigma".to_string(),
    args: vec![Expr::Integer(k), m.clone()].into(),
  }
}

/// Closed form for a pair of classified atoms, or None when the pair has
/// no identity. `with_divisor_sum` enables the `unknown * 1 -> DivisorSum`
/// rule (used on the linear-split path but not for full reduction).
fn convolve_pair(
  a: &ConvTerm,
  b: &ConvTerm,
  var: &str,
  m: &Expr,
  with_divisor_sum: bool,
) -> Option<Expr> {
  use ConvAtom::*;
  match (&a.atom, &b.atom) {
    (Power(j), Power(k)) => {
      let mn = (*j).min(*k);
      let diff = (*j - *k).abs();
      let sigma = divisor_sigma_expr(diff, m);
      if mn > 0 {
        Some(Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(m.clone()),
              right: Box::new(Expr::Integer(mn)),
            },
            sigma,
          ]
          .into(),
        })
      } else {
        Some(sigma)
      }
    }
    (Mu, Power(0)) | (Power(0), Mu) => Some(Expr::FunctionCall {
      name: "KroneckerDelta".to_string(),
      args: vec![Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          Expr::Integer(1),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), m.clone()].into(),
          },
        ]
        .into(),
      }]
      .into(),
    }),
    (Mu, Power(1)) | (Power(1), Mu) => Some(Expr::FunctionCall {
      name: "EulerPhi".to_string(),
      args: vec![m.clone()].into(),
    }),
    (Phi, Power(0)) | (Power(0), Phi) => Some(m.clone()),
    (Unknown, Power(0)) if with_divisor_sum => {
      Some(divisor_sum_expr(&a.atom_expr, var, m))
    }
    (Power(0), Unknown) if with_divisor_sum => {
      Some(divisor_sum_expr(&b.atom_expr, var, m))
    }
    _ => None,
  }
}

/// DivisorSum[m, body(#) &] with `var` replaced by the slot.
fn divisor_sum_expr(body: &Expr, var: &str, m: &Expr) -> Expr {
  let slotted = crate::syntax::substitute_variable(body, var, &Expr::Slot(1));
  let slotted = reorder_slots_last(&slotted);
  Expr::FunctionCall {
    name: "DivisorSum".to_string(),
    args: vec![
      m.clone(),
      Expr::Function {
        body: Box::new(slotted),
      },
    ]
    .into(),
  }
}

/// Wolfram's canonical order sorts bare slots after composite terms
/// (`f[#1] + #1`, `f[#1]*#1`), while Woxi's sorts them first. Reorder the
/// top-level Plus/Times factors of a function body to match, since the
/// body is held inside Function and never re-canonicalized.
fn reorder_slots_last(expr: &Expr) -> Expr {
  let is_plus = matches!(
    expr,
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      ..
    }
  ) || matches!(expr, Expr::FunctionCall { name, .. } if name == "Plus");
  let is_times = matches!(
    expr,
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      ..
    }
  ) || matches!(expr, Expr::FunctionCall { name, .. } if name == "Times");
  if !is_plus && !is_times {
    return expr.clone();
  }
  let mut parts = Vec::new();
  if is_plus {
    flatten_plus(expr, &mut parts);
    for p in parts.iter_mut() {
      *p = reorder_slots_last(p);
    }
  } else {
    flatten_times(expr, &mut parts);
  }
  let (others, slots): (Vec<Expr>, Vec<Expr>) =
    parts.into_iter().partition(|e| !matches!(e, Expr::Slot(_)));
  if slots.is_empty() {
    return expr.clone();
  }
  let mut all = others;
  all.extend(slots);
  Expr::FunctionCall {
    name: if is_plus { "Plus" } else { "Times" }.to_string(),
    args: all.into(),
  }
}

/// Multiply the two terms' coefficient factors into `core`.
fn scale_by_coeffs(core: Expr, a: &ConvTerm, b: &ConvTerm) -> Expr {
  let mut factors: Vec<Expr> = Vec::new();
  factors.extend(a.coeff.iter().cloned());
  factors.extend(b.coeff.iter().cloned());
  if factors.is_empty() {
    return core;
  }
  factors.push(core);
  Expr::FunctionCall {
    name: "Times".to_string(),
    args: factors.into(),
  }
}

pub fn dirichlet_convolve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "DirichletConvolve".to_string(),
    args: args.to_vec().into(),
  };
  let (f, g, var_expr, m) = (&args[0], &args[1], &args[2], &args[3]);
  let var = match var_expr {
    Expr::Identifier(s) => s.clone(),
    _ => return Ok(unevaluated()),
  };
  // wolframscript leaves List arguments unevaluated (no listability)
  if matches!(f, Expr::List(_)) || matches!(g, Expr::List(_)) {
    return Ok(unevaluated());
  }

  // Numeric m: positive integers expand the divisor sum directly; other
  // numeric values stay unevaluated.
  match m {
    Expr::Integer(mv) => {
      if *mv < 1 {
        return Ok(unevaluated());
      }
      let divisors = match divisors_of_i128(*mv) {
        Some(d) => d,
        None => return Ok(unevaluated()),
      };
      let terms: Vec<Expr> = divisors
        .iter()
        .map(|d| {
          let fd =
            crate::syntax::substitute_variable(f, &var, &Expr::Integer(*d));
          let gq =
            crate::syntax::substitute_variable(g, &var, &Expr::Integer(mv / d));
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![fd, gq].into(),
          }
        })
        .collect();
      return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Plus".to_string(),
        args: terms.into(),
      });
    }
    Expr::Real(_) | Expr::BigInteger(_) | Expr::BigFloat(_, _) => {
      return Ok(unevaluated());
    }
    _ => {}
  }

  // Symbolic m.
  let mut f_addends = Vec::new();
  let mut g_addends = Vec::new();
  flatten_plus(f, &mut f_addends);
  flatten_plus(g, &mut g_addends);
  let f_terms: Vec<ConvTerm> =
    f_addends.iter().map(|t| classify_term(t, &var)).collect();
  let g_terms: Vec<ConvTerm> =
    g_addends.iter().map(|t| classify_term(t, &var)).collect();

  // 1. Full reduction: every pair must have a closed form (without the
  //    DivisorSum fallback rule).
  let mut reduced: Vec<Expr> = Vec::new();
  let mut all_reduce = true;
  'outer: for ft in &f_terms {
    for gt in &g_terms {
      match convolve_pair(ft, gt, &var, m, false) {
        Some(core) => reduced.push(scale_by_coeffs(core, ft, gt)),
        None => {
          all_reduce = false;
          break 'outer;
        }
      }
    }
  }
  if all_reduce {
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Plus".to_string(),
      args: reduced.into(),
    });
  }

  // 2. g == 1: the convolution is a plain divisor sum over f (whole f,
  //    no linear split on this path — matching wolframscript).
  if matches!(g, Expr::Integer(1)) {
    return crate::evaluator::evaluate_expr_to_expr(&divisor_sum_expr(
      f, &var, m,
    ));
  }

  // 3. Linear split. wolframscript commutes the arguments first when f is
  //    a sum and g is not (so the split runs over the second argument).
  let f_is_sum = f_terms.len() > 1;
  let g_is_sum = g_terms.len() > 1;
  let (a_terms, b_terms) = if f_is_sum && !g_is_sum {
    (g_terms, f_terms)
  } else {
    (f_terms, g_terms)
  };
  let mut parts: Vec<Expr> = Vec::new();
  for at in &a_terms {
    for bt in &b_terms {
      let core = match convolve_pair(at, bt, &var, m, true) {
        Some(c) => c,
        None => Expr::FunctionCall {
          name: "DirichletConvolve".to_string(),
          args: vec![
            at.atom_expr.clone(),
            bt.atom_expr.clone(),
            Expr::Identifier(var.clone()),
            m.clone(),
          ]
          .into(),
        },
      };
      parts.push(scale_by_coeffs(core, at, bt));
    }
  }
  crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Plus".to_string(),
    args: parts.into(),
  })
}

/// All positive divisors of n in ascending order.
fn divisors_of_i128(n: i128) -> Option<Vec<i128>> {
  if n < 1 {
    return None;
  }
  let mut small = Vec::new();
  let mut large = Vec::new();
  let mut d: i128 = 1;
  while d * d <= n {
    if n % d == 0 {
      small.push(d);
      if d * d != n {
        large.push(n / d);
      }
    }
    d += 1;
  }
  large.reverse();
  small.extend(large);
  Some(small)
}
