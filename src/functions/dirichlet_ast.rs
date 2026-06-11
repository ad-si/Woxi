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
