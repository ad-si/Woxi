//! ZTransform[expr, n, z] for sequences of the form c·n^k·a^n (and a^n/n!).
//!
//! The closed forms use the Eulerian polynomials A_k:
//!   Z{n^k a^n}(z) = z·E_k(a, z) / (z − a)^(k+1)
//! with E_0 = 1, E_1 = a, E_2 = a(a + z), E_3 = a(a² + 4az + z²),
//! E_4 = a(a³ + 11a²z + 11az² + z³).
//!
//! The result is constructed structurally (not via the generic simplifier)
//! so the printed output matches wolframscript's normalization exactly:
//! integer bases use the (-a + z)^(k+1) denominator, rational bases clear
//! denominators (and flip even powers to a positive constant term, e.g.
//! (1 - 3*z)^2), and symbolic bases keep (a - z)^(k+1) with an overall
//! minus sign for even k.

use crate::InterpreterError;
use crate::functions::calculus_ast::is_constant_wrt;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator};

pub fn z_transform_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "ZTransform".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 3 {
    return Ok(unevaluated(args));
  }
  let n_var = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    _ => return Ok(unevaluated(args)),
  };
  let z_var = match &args[2] {
    Expr::Identifier(name) => name.clone(),
    _ => return Ok(unevaluated(args)),
  };
  let z = Expr::Identifier(z_var.clone());

  // Constant with respect to n: Z{c} = (c*z)/(-1 + z)
  if is_constant_wrt(&args[0], &n_var) {
    if !is_constant_wrt(&args[0], &z_var) {
      return Ok(unevaluated(args));
    }
    let num = match &args[0] {
      Expr::Integer(1) => z.clone(),
      c => times(vec![c.clone(), z.clone()]),
    };
    return Ok(divide(num, plus(vec![Expr::Integer(-1), z.clone()])));
  }

  // UnitStep[n] = 1 for n >= 0, so Z{UnitStep[n]} = z/(-1 + z).
  if matches!(&args[0], Expr::FunctionCall { name, args: fargs }
    if name == "UnitStep" && fargs.len() == 1
      && matches!(&fargs[0], Expr::Identifier(v) if *v == n_var))
  {
    return Ok(divide(z.clone(), plus(vec![Expr::Integer(-1), z.clone()])));
  }

  // Sin[a n] and Cos[a n] with a free of n and z:
  //   Z{Sin[a n]} = (z Sin[a]) / (1 + z^2 - 2 z Cos[a])
  //   Z{Cos[a n]} = (z (z - Cos[a])) / (1 + z^2 - 2 z Cos[a])
  if let Expr::FunctionCall { name, args: fargs } = &args[0]
    && (name == "Sin" || name == "Cos")
    && fargs.len() == 1
    && let Some(a) = linear_coeff_in_n(&fargs[0], &n_var)
    && is_constant_wrt(&a, &z_var)
  {
    let cos_a = func("Cos", a.clone());
    // 1 + z^2 - 2 z Cos[a]
    let den = plus(vec![
      Expr::Integer(1),
      power(z.clone(), 2),
      times(vec![Expr::Integer(-2), z.clone(), cos_a.clone()]),
    ]);
    let num = if name == "Sin" {
      times(vec![z.clone(), func("Sin", a)])
    } else {
      times(vec![
        z.clone(),
        plus(vec![z.clone(), times(vec![Expr::Integer(-1), cos_a])]),
      ])
    };
    return Ok(divide(num, den));
  }

  // Decompose the product into c * n^k * a^n (* 1/n!)
  let mut parts = TermParts::default();
  if !collect_factors(&args[0], &n_var, false, &mut parts) {
    return Ok(unevaluated(args));
  }
  let TermParts {
    k,
    c,
    base_num: p,
    base_den: q,
    sym_base,
    inv_factorial,
  } = parts;

  // a^n / n!  →  E^(a/z)
  if inv_factorial {
    if k != 0 || c != 1 || sym_base.is_some() || q != 1 || p < 1 {
      return Ok(unevaluated(args));
    }
    let exponent = if p == 1 {
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![z.clone(), Expr::Integer(-1)].into(),
      }
    } else {
      divide(Expr::Integer(p), z.clone())
    };
    return Ok(Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![Expr::Identifier("E".to_string()), exponent].into(),
    });
  }

  // Symbolic base: fixed templates for k = 0, 1, 2 (c must be 1)
  if let Some(a) = sym_base {
    if c != 1 || p != 1 || q != 1 {
      return Ok(unevaluated(args));
    }
    let a_minus_z =
      plus(vec![a.clone(), times(vec![Expr::Integer(-1), z.clone()])]);
    return Ok(match k {
      // -(z/(a - z))
      0 => negate(divide(z.clone(), a_minus_z)),
      // (a*z)/(a - z)^2
      1 => divide(times(vec![a.clone(), z.clone()]), power(a_minus_z, 2)),
      // -((a*z*(a + z))/(a - z)^3)
      2 => negate(divide(
        times(vec![a.clone(), z.clone(), plus(vec![a.clone(), z.clone()])]),
        power(a_minus_z, 3),
      )),
      _ => return Ok(unevaluated(args)),
    });
  }

  // Rational base a = p/q (positive, lowest terms), 0 <= k <= 4.
  // Numerator coefficients (ascending in z): c·q^(k+1)·E_k(p/q, z).
  if !(0..=4).contains(&k) || p < 1 || q < 1 || c < 1 {
    return Ok(unevaluated(args));
  }
  let coeffs: Vec<i128> = match k {
    0 => vec![c * q],
    1 => vec![c * p * q],
    2 => vec![c * p * p * q, c * p * q * q],
    3 => vec![c * p * p * p * q, 4 * c * p * p * q * q, c * p * q * q * q],
    4 => vec![
      c * p * p * p * p * q,
      11 * c * p * p * p * q * q,
      11 * c * p * p * q * q * q,
      c * p * q * q * q * q,
    ],
    _ => unreachable!(),
  };
  let content = coeffs.iter().fold(0i128, |acc, &x| gcd(acc, x.abs()));
  let reduced: Vec<i128> = coeffs.iter().map(|&x| x / content).collect();

  // Numerator: content * z * (poly ascending in z)
  let mut num_factors: Vec<Expr> = Vec::new();
  if content != 1 {
    num_factors.push(Expr::Integer(content));
  }
  num_factors.push(z.clone());
  if reduced != [1] {
    let mut terms: Vec<Expr> = Vec::new();
    for (i, &coef) in reduced.iter().enumerate() {
      if coef == 0 {
        continue;
      }
      let z_pow = match i {
        0 => None,
        1 => Some(z.clone()),
        _ => Some(power(z.clone(), i as i128)),
      };
      terms.push(match (coef, z_pow) {
        (cf, None) => Expr::Integer(cf),
        (1, Some(zp)) => zp,
        (cf, Some(zp)) => times(vec![Expr::Integer(cf), zp]),
      });
    }
    num_factors.push(plus(terms));
  }
  let num = if num_factors.len() == 1 {
    num_factors.remove(0)
  } else {
    times(num_factors)
  };

  // Denominator: (−p + q·z)^(k+1); even powers with q > 1 flip to the
  // positive-constant form (p − q·z)^(k+1)
  let qz = if q == 1 {
    z.clone()
  } else {
    times(vec![Expr::Integer(q), z.clone()])
  };
  let base = if (k + 1) % 2 == 0 && q > 1 {
    plus(vec![
      Expr::Integer(p),
      times(vec![Expr::Integer(-q), z.clone()]),
    ])
  } else {
    plus(vec![Expr::Integer(-p), qz])
  };
  let den = if k == 0 { base } else { power(base, k + 1) };

  Ok(divide(num, den))
}

#[derive(Default)]
struct TermParts {
  k: i128,
  c: i128,
  base_num: i128,
  base_den: i128,
  sym_base: Option<Expr>,
  inv_factorial: bool,
}

impl TermParts {
  fn init(&mut self) {
    if self.c == 0 {
      self.c = 1;
      self.base_num = 1;
      self.base_den = 1;
    }
  }
}

/// Flatten the product structure of `expr` into TermParts.
/// `inverted` flags factors appearing in a denominator.
/// Returns false for any factor outside the supported c·n^k·a^n/n! shape.
fn collect_factors(
  expr: &Expr,
  n_var: &str,
  inverted: bool,
  parts: &mut TermParts,
) -> bool {
  parts.init();
  match expr {
    Expr::FunctionCall { name, args } if name == "Times" => args
      .iter()
      .all(|a| collect_factors(a, n_var, inverted, parts)),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      collect_factors(left, n_var, inverted, parts)
        && collect_factors(right, n_var, inverted, parts)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      collect_factors(left, n_var, inverted, parts)
        && collect_factors(right, n_var, !inverted, parts)
    }
    Expr::Identifier(name) if name == n_var => {
      if inverted {
        return false;
      }
      parts.k += 1;
      true
    }
    Expr::Integer(m) if *m >= 1 => {
      if inverted {
        return false;
      }
      parts.c *= m;
      true
    }
    _ => {
      if let Some((base, exp)) = as_power(expr) {
        // n^k
        if matches!(&base, Expr::Identifier(b) if b == n_var) {
          if let Expr::Integer(kk) = exp
            && kk >= 1
            && !inverted
          {
            parts.k += kk;
            return true;
          }
          return false;
        }
        // (n!)^(-1)
        if let Expr::FunctionCall { name, args } = &base
          && name == "Factorial"
          && args.len() == 1
          && matches!(&args[0], Expr::Identifier(b) if b == n_var)
          && matches!(exp, Expr::Integer(-1))
          && !inverted
        {
          parts.inv_factorial = true;
          return true;
        }
        // a^n / a^(-n)
        let flip = match &exp {
          Expr::Identifier(e) if e == n_var => false,
          Expr::FunctionCall { name, args }
            if name == "Times"
              && args.len() == 2
              && matches!(&args[0], Expr::Integer(-1))
              && matches!(&args[1], Expr::Identifier(e) if e == n_var) =>
          {
            true
          }
          Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand,
          } if matches!(operand.as_ref(), Expr::Identifier(e) if e == n_var) => {
            true
          }
          _ => return false,
        };
        let inv = inverted ^ flip;
        match &base {
          Expr::Integer(b) if *b >= 1 => {
            if inv {
              parts.base_den *= b;
            } else {
              parts.base_num *= b;
            }
            reduce_base(parts);
            true
          }
          Expr::FunctionCall { name, args }
            if name == "Rational" && args.len() == 2 =>
          {
            if let (Expr::Integer(bn), Expr::Integer(bd)) = (&args[0], &args[1])
              && *bn >= 1
              && *bd >= 1
            {
              if inv {
                parts.base_num *= bd;
                parts.base_den *= bn;
              } else {
                parts.base_num *= bn;
                parts.base_den *= bd;
              }
              reduce_base(parts);
              true
            } else {
              false
            }
          }
          Expr::Identifier(s) if s != n_var && !inv => {
            if parts.sym_base.is_some() {
              return false;
            }
            parts.sym_base = Some(Expr::Identifier(s.clone()));
            true
          }
          _ => false,
        }
      } else if let Expr::FunctionCall { name, args } = expr
        && name == "Factorial"
        && args.len() == 1
        && matches!(&args[0], Expr::Identifier(b) if b == n_var)
        && inverted
      {
        parts.inv_factorial = true;
        true
      } else {
        false
      }
    }
  }
}

fn reduce_base(parts: &mut TermParts) {
  let g = gcd(parts.base_num, parts.base_den);
  if g > 1 {
    parts.base_num /= g;
    parts.base_den /= g;
  }
}

fn as_power(expr: &Expr) -> Option<(Expr, Expr)> {
  match expr {
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      Some((args[0].clone(), args[1].clone()))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => Some(((**left).clone(), (**right).clone())),
    _ => None,
  }
}

fn gcd(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

fn plus(terms: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.into(),
  }
}

fn times(factors: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: "Times".to_string(),
    args: factors.into(),
  }
}

fn divide(num: Expr, den: Expr) -> Expr {
  Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(num),
    right: Box::new(den),
  }
}

fn power(base: Expr, exp: i128) -> Expr {
  Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(base),
    right: Box::new(Expr::Integer(exp)),
  }
}

fn negate(e: Expr) -> Expr {
  Expr::UnaryOp {
    op: UnaryOperator::Minus,
    operand: Box::new(e),
  }
}

fn func(name: &str, arg: Expr) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: vec![arg].into(),
  }
}

/// If `arg` is exactly `a * n` — degree one in `n` with no constant term and
/// `a` free of `n` — return `a` (a plain `n` gives `a = 1`). Phase-shifted or
/// higher-degree arguments return None.
fn linear_coeff_in_n(arg: &Expr, n_var: &str) -> Option<Expr> {
  if matches!(arg, Expr::Identifier(v) if v == n_var) {
    return Some(Expr::Integer(1));
  }
  let mut n_count = 0;
  let mut rest: Vec<Expr> = Vec::new();
  for f in times_factors(arg) {
    if matches!(&f, Expr::Identifier(v) if v == n_var) {
      n_count += 1;
    } else if is_constant_wrt(&f, n_var) {
      rest.push(f);
    } else {
      return None;
    }
  }
  if n_count != 1 {
    return None;
  }
  Some(match rest.len() {
    0 => Expr::Integer(1),
    1 => rest.remove(0),
    _ => times(rest),
  })
}

// ─── InverseZTransform ───────────────────────────────────────────────

type Frac = (i128, i128); // (numerator, denominator), denominator > 0

fn frac(n: i128, d: i128) -> Frac {
  let g = gcd(n, d);
  let (mut n, mut d) = if g == 0 { (0, 1) } else { (n / g, d / g) };
  if d < 0 {
    n = -n;
    d = -d;
  }
  (n, d)
}

fn frac_add(a: Frac, b: Frac) -> Frac {
  frac(a.0 * b.1 + b.0 * a.1, a.1 * b.1)
}

fn frac_mul(a: Frac, b: Frac) -> Frac {
  frac(a.0 * b.0, a.1 * b.1)
}

fn frac_div(a: Frac, b: Frac) -> Option<Frac> {
  if b.0 == 0 {
    return None;
  }
  Some(frac(a.0 * b.1, a.1 * b.0))
}

/// Parse `expr` as a polynomial in `var` with rational coefficients.
/// Returns ascending coefficients, or None for non-polynomial input.
fn poly_coeffs(expr: &Expr, var: &str) -> Option<Vec<Frac>> {
  match expr {
    Expr::Integer(m) => Some(vec![(*m, 1)]),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(a), Expr::Integer(b)) = (&args[0], &args[1]) {
        Some(vec![frac(*a, *b)])
      } else {
        None
      }
    }
    Expr::Identifier(v) if v == var => Some(vec![(0, 1), (1, 1)]),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Some(poly_neg(&poly_coeffs(operand, var)?)),
    Expr::FunctionCall { name, args } if name == "Plus" => {
      let mut acc = vec![(0i128, 1i128)];
      for a in args {
        acc = poly_add(&acc, &poly_coeffs(a, var)?);
      }
      Some(acc)
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut acc = vec![(1i128, 1i128)];
      for a in args {
        acc = poly_mul(&acc, &poly_coeffs(a, var)?);
      }
      Some(acc)
    }
    Expr::BinaryOp { op, left, right } => match op {
      BinaryOperator::Plus => Some(poly_add(
        &poly_coeffs(left, var)?,
        &poly_coeffs(right, var)?,
      )),
      BinaryOperator::Minus => Some(poly_add(
        &poly_coeffs(left, var)?,
        &poly_neg(&poly_coeffs(right, var)?),
      )),
      BinaryOperator::Times => Some(poly_mul(
        &poly_coeffs(left, var)?,
        &poly_coeffs(right, var)?,
      )),
      BinaryOperator::Power => {
        if let Expr::Integer(k) = right.as_ref()
          && *k >= 1
        {
          let base = poly_coeffs(left, var)?;
          let mut acc = vec![(1i128, 1i128)];
          for _ in 0..*k {
            acc = poly_mul(&acc, &base);
          }
          Some(acc)
        } else {
          None
        }
      }
      _ => None,
    },
    _ => None,
  }
}

fn poly_neg(p: &[Frac]) -> Vec<Frac> {
  p.iter().map(|&(n, d)| (-n, d)).collect()
}

fn poly_add(a: &[Frac], b: &[Frac]) -> Vec<Frac> {
  let len = a.len().max(b.len());
  (0..len)
    .map(|i| {
      frac_add(*a.get(i).unwrap_or(&(0, 1)), *b.get(i).unwrap_or(&(0, 1)))
    })
    .collect()
}

fn poly_mul(a: &[Frac], b: &[Frac]) -> Vec<Frac> {
  let mut out = vec![(0i128, 1i128); a.len() + b.len() - 1];
  for (i, &ca) in a.iter().enumerate() {
    for (j, &cb) in b.iter().enumerate() {
      out[i + j] = frac_add(out[i + j], frac_mul(ca, cb));
    }
  }
  out
}

fn poly_trim(mut p: Vec<Frac>) -> Vec<Frac> {
  while p.len() > 1 && p.last() == Some(&(0, 1)) {
    p.pop();
  }
  p
}

/// Split expr into (sign, numerator, denominator) structurally.
/// Handles the evaluator's canonical Times[..., Power[den, -m]] form as
/// well as explicit Divide and unary minus.
fn split_fraction(expr: &Expr) -> (i128, Expr, Expr) {
  match expr {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let (s, n, d) = split_fraction(operand);
      (-s, n, d)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      let (s, n, _) = split_fraction(left);
      (s, n, (**right).clone())
    }
    _ => {
      let mut sign = 1i128;
      let mut num: Vec<Expr> = Vec::new();
      let mut den: Vec<Expr> = Vec::new();
      for f in times_factors(expr) {
        if matches!(f, Expr::Integer(-1)) {
          sign = -sign;
          continue;
        }
        if let Some((base, exp)) = as_power(&f)
          && let Expr::Integer(e) = exp
          && e < 0
        {
          den.push(if e == -1 {
            base
          } else {
            Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(base),
              right: Box::new(Expr::Integer(-e)),
            }
          });
          continue;
        }
        num.push(f);
      }
      let num_expr = match num.len() {
        0 => Expr::Integer(1),
        1 => num.remove(0),
        _ => times(num),
      };
      let den_expr = match den.len() {
        0 => Expr::Integer(1),
        1 => den.remove(0),
        _ => times(den),
      };
      (sign, num_expr, den_expr)
    }
  }
}

/// InverseZTransform[expr, z, n] — inverse of the sequence shapes that
/// ZTransform produces: c·n^k·a^n monomial sequences, Binomial-type
/// z/(z-1)^m inputs, and exponential E^(a/z) forms.
pub fn inverse_z_transform_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "InverseZTransform".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 3 {
    return Ok(unevaluated(args));
  }
  let z_var = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    _ => return Ok(unevaluated(args)),
  };
  let n_var = match &args[2] {
    Expr::Identifier(name) => name.clone(),
    _ => return Ok(unevaluated(args)),
  };
  let n = Expr::Identifier(n_var.clone());
  let factorial_n = Expr::FunctionCall {
    name: "Factorial".to_string(),
    args: vec![n.clone()].into(),
  };

  // E^(1/z) → 1/n!, E^(c/z) → c^n/n!
  if let Some((base, exp)) = as_power(&args[0])
    && (matches!(&base, Expr::Identifier(e) if e == "E")
      || matches!(&base, Expr::Constant(e) if e == "E"))
  {
    // exponent 1/z
    let c: Option<i128> = match &exp {
      _ if matches!(as_power(&exp), Some((b, Expr::Integer(-1)))
        if matches!(&b, Expr::Identifier(v) if *v == z_var)) =>
      {
        Some(1)
      }
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left,
        right,
      } if matches!(right.as_ref(), Expr::Identifier(v) if *v == z_var) => {
        match left.as_ref() {
          Expr::Integer(c) if *c >= 1 => Some(*c),
          _ => None,
        }
      }
      Expr::FunctionCall { name, args: targs }
        if name == "Times" && targs.len() == 2 =>
      {
        match (&targs[0], as_power(&targs[1])) {
          (Expr::Integer(c), Some((b, Expr::Integer(-1))))
            if *c >= 1 && matches!(&b, Expr::Identifier(v) if *v == z_var) =>
          {
            Some(*c)
          }
          _ => None,
        }
      }
      _ => None,
    };
    return Ok(match c {
      Some(1) => Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(factorial_n),
        right: Box::new(Expr::Integer(-1)),
      },
      Some(c) => divide(
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(Expr::Integer(c)),
          right: Box::new(n.clone()),
        },
        factorial_n,
      ),
      None => unevaluated(args),
    });
  }

  // Constant with respect to z: c → c*DiscreteDelta[n]
  if is_constant_wrt(&args[0], &z_var) {
    let delta = Expr::FunctionCall {
      name: "DiscreteDelta".to_string(),
      args: vec![n.clone()].into(),
    };
    return Ok(match &args[0] {
      Expr::Integer(1) => delta,
      c => times(vec![c.clone(), delta]),
    });
  }

  let (mut sign, num, den) = split_fraction(&args[0]);
  if matches!(den, Expr::Integer(1)) {
    return Ok(unevaluated(args));
  }

  // Denominator: linear^m
  let (den_base, m) = match as_power(&den) {
    Some((b, Expr::Integer(m))) if (2..=5).contains(&m) => (b, m as usize),
    Some(_) => return Ok(unevaluated(args)),
    None => (den.clone(), 1),
  };

  // Symbolic base: (a - z)^m or (z - a)^m with a an atom
  if let Some(a) = symbolic_linear_root(&den_base, &z_var) {
    // (z - a)^m = (-1)^m (a - z)^m
    let norm_sign = if matches!(&den_base, Expr::FunctionCall { name, args: pargs }
      if name == "Plus" && pargs.len() == 2
        && matches!(&pargs[1], Expr::Identifier(v) if *v == z_var))
    {
      // base is (-a + z)
      if m % 2 == 1 { -sign } else { sign }
    } else {
      sign
    };
    sign = norm_sign;
    let a_pow_n = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(a.clone()),
      right: Box::new(n.clone()),
    };
    let factors = times_factors(&num);
    let fstr: Vec<String> =
      factors.iter().map(crate::syntax::expr_to_string).collect();
    let a_str = crate::syntax::expr_to_string(&a);
    let matches_multiset = |expected: &[String]| {
      let mut want = expected.to_vec();
      want.sort();
      let mut got = fstr.clone();
      got.sort();
      want == got
    };
    // -(z/(a - z)) → a^n
    if m == 1 && sign == -1 && matches_multiset(&[z_var.clone()]) {
      return Ok(a_pow_n);
    }
    // (a*z)/(a - z)^2 → a^n*n
    if m == 2 && sign == 1 && matches_multiset(&[a_str.clone(), z_var.clone()])
    {
      return Ok(times(vec![a_pow_n, n.clone()]));
    }
    // -((a*z*(a + z))/(a - z)^3) → a^n*n^2
    if m == 3
      && sign == -1
      && matches_multiset(&[
        a_str.clone(),
        z_var.clone(),
        crate::syntax::expr_to_string(&plus(vec![
          a.clone(),
          Expr::Identifier(z_var.clone()),
        ])),
      ])
    {
      return Ok(times(vec![
        a_pow_n,
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(n.clone()),
          right: Box::new(Expr::Integer(2)),
        },
      ]));
    }
    return Ok(unevaluated(args));
  }

  // Constant rule: (w*z)/(-1 + z) with w free of z (e.g. symbol x)
  // handled before requiring rational coefficients
  let den_lin = poly_coeffs(&den_base, &z_var);
  if m == 1
    && den_lin == Some(vec![(-1, 1), (1, 1)])
    && let factors = times_factors(&num)
    && factors
      .iter()
      .filter(|f| matches!(f, Expr::Identifier(v) if **v == *z_var))
      .count()
      == 1
  {
    let rest: Vec<Expr> = factors
      .iter()
      .filter(|f| !matches!(f, Expr::Identifier(v) if **v == *z_var))
      .cloned()
      .collect();
    let all_free = rest
      .iter()
      .all(|f| is_constant_wrt(f, &z_var) && is_constant_wrt(f, &n_var));
    if all_free && !rest.iter().any(|f| poly_coeffs(f, &z_var).is_some()) {
      let w = match rest.len() {
        0 => Expr::Integer(1),
        1 => rest.into_iter().next().unwrap(),
        _ => times(rest),
      };
      return Ok(if sign == -1 { negate(w) } else { w });
    }
  }

  // Numeric path: rational-coefficient numerator and linear denominator
  let den_lin = match den_lin {
    Some(p) if poly_trim(p.clone()).len() == 2 => poly_trim(p),
    _ => return Ok(unevaluated(args)),
  };
  let (alpha, beta) = (den_lin[0], den_lin[1]);
  let num_poly = match poly_coeffs(&num, &z_var) {
    Some(p) => poly_trim(p),
    None => return Ok(unevaluated(args)),
  };
  // numerator must be z·Q(z) with deg Q ≤ m−1
  if num_poly[0] != (0, 1) || num_poly.len() - 1 > m {
    return Ok(unevaluated(args));
  }
  let q_poly: Vec<Frac> = num_poly[1..].to_vec();

  // root r = -alpha/beta = p/q (positive only)
  let r = match frac_div((-alpha.0, alpha.1), beta) {
    Some(r) if r.0 >= 1 => r,
    _ => return Ok(unevaluated(args)),
  };
  let (p, q) = r;
  // lambda = beta/q: den = lambda^m * (-p + q*z)^m
  let lambda = frac(beta.0, beta.1 * q);

  // Forward basis: Z{n^j r^n} = z*C_j(z)/(-p + q*z)^(j+1)
  let basis: Vec<Vec<i128>> = vec![
    vec![q],
    vec![p * q],
    vec![p * p * q, p * q * q],
    vec![p * p * p * q, 4 * p * p * q * q, p * q * q * q],
    vec![
      p * p * p * p * q,
      11 * p * p * p * q * q,
      11 * p * p * q * q * q,
      p * q * q * q * q,
    ],
  ];

  // (-p + q*z)^e coefficients
  let lin_pow = |e: usize| -> Vec<Frac> {
    let mut acc = vec![(1i128, 1i128)];
    for _ in 0..e {
      acc = poly_mul(&acc, &[(-p, 1), (q, 1)]);
    }
    acc
  };

  // Solve sum_j d_j * C_j(z) * (-p+q*z)^(m-1-j) * lambda^m = sign * Q(z)
  let mut matrix: Vec<Vec<Frac>> = vec![vec![(0, 1); m]; m]; // [row=z power][col=j]
  for (j, c_j) in basis.iter().enumerate().take(m) {
    let cj: Vec<Frac> = c_j.iter().map(|&x| (x, 1)).collect();
    let col = poly_mul(&cj, &lin_pow(m - 1 - j));
    for (i, &coef) in col.iter().enumerate() {
      matrix[i][j] = coef;
    }
  }
  let mut lambda_m = (1i128, 1i128);
  for _ in 0..m {
    lambda_m = frac_mul(lambda_m, lambda);
  }
  let mut rhs: Vec<Frac> = (0..m)
    .map(|i| {
      let qc = *q_poly.get(i).unwrap_or(&(0, 1));
      frac_div(frac((sign * qc.0, qc.1).0, qc.1), lambda_m).unwrap_or((0, 1))
    })
    .collect();

  // Gaussian elimination (exact rational, m ≤ 5)
  let mut d = vec![(0i128, 1i128); m];
  {
    let mut a = matrix.clone();
    for col in 0..m {
      let pivot = (col..m).find(|&r0| a[r0][col].0 != 0);
      let pivot = match pivot {
        Some(p0) => p0,
        None => return Ok(unevaluated(args)),
      };
      a.swap(col, pivot);
      rhs.swap(col, pivot);
      for row in (col + 1)..m {
        let factor = match frac_div(a[row][col], a[col][col]) {
          Some(f) => f,
          None => return Ok(unevaluated(args)),
        };
        for kk in col..m {
          let sub = frac_mul(factor, a[col][kk]);
          a[row][kk] = frac_add(a[row][kk], (-sub.0, sub.1));
        }
        let sub = frac_mul(factor, rhs[col]);
        rhs[row] = frac_add(rhs[row], (-sub.0, sub.1));
      }
    }
    for i in (0..m).rev() {
      let mut acc = rhs[i];
      for j in (i + 1)..m {
        let sub = frac_mul(matrix_at(&a, i, j), d[j]);
        acc = frac_add(acc, (-sub.0, sub.1));
      }
      d[i] = match frac_div(acc, a[i][i]) {
        Some(v) => v,
        None => return Ok(unevaluated(args)),
      };
    }
  }

  format_inverse_result(&d, p, q, m, &n, &n_var)
    .map(|opt| opt.unwrap_or_else(|| unevaluated(args)))
}

fn matrix_at(a: &[Vec<Frac>], i: usize, j: usize) -> Frac {
  a[i][j]
}

/// Format sum_j d_j n^j p/q^n in wolframscript's style; None if the
/// coefficient vector is outside the verified output inventory.
fn format_inverse_result(
  d: &[Frac],
  p: i128,
  q: i128,
  m: usize,
  n: &Expr,
  _n_var: &str,
) -> Result<Option<Expr>, InterpreterError> {
  let nonzero: Vec<usize> = d
    .iter()
    .enumerate()
    .filter(|&(_, &c)| c.0 != 0)
    .map(|(i, _)| i)
    .collect();
  if nonzero.is_empty() {
    return Ok(Some(Expr::Integer(0)));
  }

  let power = |b: Expr, e: Expr| Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(b),
    right: Box::new(e),
  };
  let n_pow = |j: usize| match j {
    1 => n.clone(),
    _ => power(n.clone(), Expr::Integer(j as i128)),
  };

  // Single monomial d_J * n^J * r^n
  if nonzero.len() == 1 {
    let j = nonzero[0];
    let coef = d[j];
    if coef == (1, 1) {
      // base r = p/q
      return Ok(Some(match (p, q, j) {
        (1, 1, 0) => Expr::Integer(1),
        (1, 1, _) => n_pow(j),
        (_, 1, 0) => power(Expr::Integer(p), n.clone()),
        (_, 1, _) => times(vec![power(Expr::Integer(p), n.clone()), n_pow(j)]),
        (1, _, 0) => power(Expr::Integer(q), negate(n.clone())),
        (1, _, _) => divide(n_pow(j), power(Expr::Integer(q), n.clone())),
        _ => return Ok(None),
      }));
    }
    // Constant sequence c (r = 1, j = 0)
    if j == 0 && p == 1 && q == 1 {
      return Ok(Some(if coef.1 == 1 {
        Expr::Integer(coef.0)
      } else {
        Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(coef.0), Expr::Integer(coef.1)].into(),
        }
      }));
    }
    return Ok(None);
  }

  // Binomial[n, m-1] for z/(z-1)^m: d must equal the coefficients of
  // n(n-1)...(n-m+2)/(m-1)!
  if p == 1 && q == 1 && m >= 3 {
    let mut fall = vec![(0i128, 1i128), (1, 1)]; // n
    for i in 1..(m - 1) as i128 {
      fall = poly_mul(&fall, &[(-i, 1), (1, 1)]);
    }
    let mut fact = 1i128;
    for i in 2..(m as i128) {
      fact *= i;
    }
    let expected: Vec<Frac> =
      fall.iter().map(|&(a, b)| frac(a, b * fact)).collect();
    let mut padded = d.to_vec();
    while padded.len() < expected.len() {
      padded.push((0, 1));
    }
    if padded == expected {
      // ((-(m-2) + n)*...*(-1 + n)*n)/(m-1)!
      let mut factors: Vec<Expr> = Vec::new();
      for i in (1..=(m - 2) as i128).rev() {
        factors.push(plus(vec![Expr::Integer(-i), n.clone()]));
      }
      factors.push(n.clone());
      return Ok(Some(divide(times(factors), Expr::Integer(fact))));
    }
  }

  Ok(None)
}

/// If `expr` is linear in z with a symbolic atom root — (a - z) or
/// (-a + z) with `a` an identifier — return a.
fn symbolic_linear_root(expr: &Expr, z_var: &str) -> Option<Expr> {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Plus"
    && args.len() == 2
  {
    // (a - z): Plus[a, Times[-1, z]]
    if let Expr::Identifier(a) = &args[0]
      && a != z_var
      && matches!(&args[1], Expr::FunctionCall { name: tn, args: targs }
        if tn == "Times" && targs.len() == 2
          && matches!(&targs[0], Expr::Integer(-1))
          && matches!(&targs[1], Expr::Identifier(v) if v == z_var))
    {
      return Some(Expr::Identifier(a.clone()));
    }
    // (-a + z): Plus[Times[-1, a], z]
    if matches!(&args[1], Expr::Identifier(v) if v == z_var)
      && let Expr::FunctionCall {
        name: tn,
        args: targs,
      } = &args[0]
      && tn == "Times"
      && targs.len() == 2
      && matches!(&targs[0], Expr::Integer(-1))
      && matches!(&targs[1], Expr::Identifier(a) if a != z_var)
      && let Expr::Identifier(a) = &targs[1]
    {
      return Some(Expr::Identifier(a.clone()));
    }
  }
  None
}

/// Flatten Times into a factor list (single non-Times expr → one factor).
fn times_factors(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::FunctionCall { name, args } if name == "Times" => {
      args.iter().flat_map(times_factors).collect()
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let mut out = times_factors(left);
      out.extend(times_factors(right));
      out
    }
    _ => vec![expr.clone()],
  }
}

// ─── FourierCoefficient ──────────────────────────────────────────────

/// FourierCoefficient[f, t, n] for polynomials f of degree <= 3 with
/// rational coefficients, in wolframscript's closed forms:
/// the n = 0 piece carries the mean (constants and Pi^2/3 per t^2),
/// the general piece sums the per-monomial terms (I*(-1)^n)/n,
/// (2*(-1)^n)/n^2, (I*(-1)^n*(-6 + n^2*Pi^2))/n^3; a pure constant c
/// gives c*DiscreteDelta[n].
pub fn fourier_coefficient_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "FourierCoefficient".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 3 {
    return Ok(unevaluated(args));
  }
  let t_var = match &args[1] {
    Expr::Identifier(v) => v.clone(),
    _ => return Ok(unevaluated(args)),
  };
  let n_arg = args[2].clone();

  // Polynomial in t with rational coefficients, degree <= 3
  let coeffs = match poly_coeffs(&args[0], &t_var) {
    Some(c) => c,
    None => return Ok(unevaluated(args)),
  };
  if coeffs.len() > 4 {
    return Ok(unevaluated(args));
  }
  let coeff = |k: usize| *coeffs.get(k).unwrap_or(&(0, 1));
  let frac_expr = |f: Frac| -> Expr {
    let f = frac(f.0, f.1);
    if f.1 == 1 {
      Expr::Integer(f.0)
    } else {
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(f.0), Expr::Integer(f.1)].into(),
      }
    }
  };
  let times = |fs: Vec<Expr>| Expr::FunctionCall {
    name: "Times".to_string(),
    args: fs.into(),
  };
  let div = |a: Expr, b: Expr| Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(a),
    right: Box::new(b),
  };
  let pow = |b: Expr, e: i128| Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(b),
    right: Box::new(Expr::Integer(e)),
  };
  let i_unit = || Expr::Identifier("I".to_string());
  let pi = || Expr::Constant("Pi".to_string());

  let (c0, c1, c2, c3) = (coeff(0), coeff(1), coeff(2), coeff(3));

  // Pure constant: c*DiscreteDelta[n]
  if c1.0 == 0 && c2.0 == 0 && c3.0 == 0 {
    let delta = Expr::FunctionCall {
      name: "DiscreteDelta".to_string(),
      args: vec![n_arg.clone()].into(),
    };
    let result = if c0 == (1, 1) {
      delta
    } else {
      times(vec![frac_expr(c0), delta])
    };
    return crate::evaluator::evaluate_expr_to_expr(&result);
  }

  // n == 0 piece: c0 + c2*Pi^2/3 (odd powers integrate to zero)
  let zero_piece: Expr = {
    let mut terms: Vec<Expr> = Vec::new();
    if c0.0 != 0 {
      terms.push(frac_expr(c0));
    }
    if c2.0 != 0 {
      let scaled = frac(c2.0, c2.1 * 3);
      terms.push(if scaled == (1, 1) {
        pow(pi(), 2)
      } else {
        // Pi^2/3-style quotient (or c*Pi^2 for integer multiples)
        match scaled {
          (p, 1) => times(vec![Expr::Integer(p), pow(pi(), 2)]),
          (1, q) => div(pow(pi(), 2), Expr::Integer(q)),
          (p, q) => div(
            times(vec![Expr::Integer(p), pow(pi(), 2)]),
            Expr::Integer(q),
          ),
        }
      });
    }
    match terms.len() {
      0 => Expr::Integer(0),
      1 => terms.remove(0),
      _ => Expr::FunctionCall {
        name: "Plus".to_string(),
        args: terms.into(),
      },
    }
  };

  // General piece: sum of the per-monomial closed forms.
  // Coefficient grouping: rational multiples of I print as (2*I).
  let coeff_i = |f: Frac| -> Option<Expr> {
    match frac(f.0, f.1) {
      (0, _) => None,
      (1, 1) => Some(i_unit()),
      (p, 1) => Some(times(vec![Expr::Integer(p), i_unit()])),
      (p, q) => Some(div(
        times(vec![Expr::Integer(p), i_unit()]),
        Expr::Integer(q),
      )),
    }
  };
  let m1n = || Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(Expr::Integer(-1)),
    right: Box::new(n_arg.clone()),
  };
  let n_expr = || n_arg.clone();

  let mut general_terms: Vec<Expr> = Vec::new();
  // t: (c1*I*(-1)^n)/n
  if let Some(ci) = coeff_i(c1) {
    general_terms.push(div(times(vec![ci, m1n()]), n_expr()));
  }
  // t^2: (2*c2*(-1)^n)/n^2
  if c2.0 != 0 {
    let f = frac(2 * c2.0, c2.1);
    let lead = frac_expr(f);
    general_terms.push(div(times(vec![lead, m1n()]), pow(n_expr(), 2)));
  }
  // t^3: (c3*I*(-1)^n*(-6 + n^2*Pi^2))/n^3
  if let Some(ci) = coeff_i(c3) {
    let bracket = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        Expr::Integer(-6),
        times(vec![pow(n_expr(), 2), pow(pi(), 2)]),
      ]
      .into(),
    };
    general_terms.push(div(times(vec![ci, m1n(), bracket]), pow(n_expr(), 3)));
  }
  let general = match general_terms.len() {
    0 => Expr::Integer(0),
    1 => general_terms.remove(0),
    _ => Expr::FunctionCall {
      name: "Plus".to_string(),
      args: general_terms.into(),
    },
  };

  // Numeric n: pick the branch and evaluate
  if let Expr::Integer(n_val) = &n_arg {
    let chosen = if *n_val == 0 { zero_piece } else { general };
    return crate::evaluator::evaluate_expr_to_expr(&chosen);
  }
  if !matches!(&n_arg, Expr::Identifier(_)) {
    return Ok(unevaluated(args));
  }

  // Symbolic n: Piecewise[{{zero_piece, n == 0}}, general]
  let cond = Expr::Comparison {
    operands: vec![n_arg.clone(), Expr::Integer(0)],
    operators: vec![crate::syntax::ComparisonOp::Equal],
  };
  Ok(Expr::FunctionCall {
    name: "Piecewise".to_string(),
    args: vec![
      Expr::List(vec![Expr::List(vec![zero_piece, cond].into())].into()),
      general,
    ]
    .into(),
  })
}

// ─── FourierSinCoefficient / FourierCosCoefficient ───────────────────

/// Half-range Fourier sine/cosine coefficients over (0, Pi) for single
/// monomials c*t^k (k <= 3), in wolframscript's closed forms.
pub fn fourier_sin_cos_coefficient_ast(
  args: &[Expr],
  sine: bool,
) -> Result<Expr, InterpreterError> {
  let head = if sine {
    "FourierSinCoefficient"
  } else {
    "FourierCosCoefficient"
  };
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: head.to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 3 {
    return Ok(unevaluated(args));
  }
  let t_var = match &args[1] {
    Expr::Identifier(v) => v.clone(),
    _ => return Ok(unevaluated(args)),
  };
  let n_arg = args[2].clone();
  if !matches!(&n_arg, Expr::Identifier(_) | Expr::Integer(_)) {
    return Ok(unevaluated(args));
  }

  // Single monomial c*t^k with rational c, k <= 3
  let coeffs = match poly_coeffs(&args[0], &t_var) {
    Some(c) => c,
    None => return Ok(unevaluated(args)),
  };
  if coeffs.len() > 4 {
    return Ok(unevaluated(args));
  }
  let nonzero: Vec<(usize, Frac)> = coeffs
    .iter()
    .enumerate()
    .filter(|(_, f)| f.0 != 0)
    .map(|(k, f)| (k, *f))
    .collect();
  if nonzero.len() != 1 {
    return Ok(unevaluated(args));
  }
  let (k, c) = nonzero[0];

  let times = |fs: Vec<Expr>| Expr::FunctionCall {
    name: "Times".to_string(),
    args: fs.into(),
  };
  let plus = |ts: Vec<Expr>| Expr::FunctionCall {
    name: "Plus".to_string(),
    args: ts.into(),
  };
  let div = |a: Expr, b: Expr| Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(a),
    right: Box::new(b),
  };
  let pow = |b: Expr, e: i128| Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(b),
    right: Box::new(Expr::Integer(e)),
  };
  let pi = || Expr::Constant("Pi".to_string());
  let n_e = || n_arg.clone();
  let m1 = || Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(Expr::Integer(-1)),
    right: Box::new(n_arg.clone()),
  };
  let scaled = |base: i128| -> Expr {
    let f = frac(base * c.0, c.1);
    if f.1 == 1 {
      Expr::Integer(f.0)
    } else {
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(f.0), Expr::Integer(f.1)].into(),
      }
    }
  };
  // 2 - 2*(-1)^n + (-1)^n*n^2*Pi^2
  let bracket2 = || {
    plus(vec![
      Expr::Integer(2),
      times(vec![Expr::Integer(-2), m1()]),
      times(vec![m1(), pow(n_e(), 2), pow(pi(), 2)]),
    ])
  };

  // Cosine at n = 0: the mean integral (2/Pi) Integrate[f, {t, 0, Pi}]
  if !sine && matches!(&n_arg, Expr::Integer(0)) {
    let value = match k {
      0 => scaled(2),
      1 => times(vec![scaled(1), pi()]),
      2 => div(times(vec![scaled(2), pow(pi(), 2)]), Expr::Integer(3)),
      3 => div(times(vec![scaled(1), pow(pi(), 3)]), Expr::Integer(2)),
      _ => return Ok(unevaluated(args)),
    };
    return crate::evaluator::evaluate_expr_to_expr(&value);
  }

  let general: Expr = if sine {
    match k {
      // (-2*(-1 + (-1)^n))/(n*Pi)
      0 => div(
        times(vec![scaled(-2), plus(vec![Expr::Integer(-1), m1()])]),
        times(vec![n_e(), pi()]),
      ),
      // (-2*(-1)^n)/n
      1 => div(times(vec![scaled(-2), m1()]), n_e()),
      // (-2*(2 - 2*(-1)^n + (-1)^n*n^2*Pi^2))/(n^3*Pi)
      2 => div(
        times(vec![scaled(-2), bracket2()]),
        times(vec![pow(n_e(), 3), pi()]),
      ),
      // (-2*(-1)^n*(-6 + n^2*Pi^2))/n^3
      3 => div(
        times(vec![
          scaled(-2),
          m1(),
          plus(vec![
            Expr::Integer(-6),
            times(vec![pow(n_e(), 2), pow(pi(), 2)]),
          ]),
        ]),
        pow(n_e(), 3),
      ),
      _ => return Ok(unevaluated(args)),
    }
  } else {
    match k {
      // 2*DiscreteDelta[n]
      0 => times(vec![
        scaled(2),
        Expr::FunctionCall {
          name: "DiscreteDelta".to_string(),
          args: vec![n_e()].into(),
        },
      ]),
      // (2*(-1 + (-1)^n))/(n^2*Pi)
      1 => div(
        times(vec![scaled(2), plus(vec![Expr::Integer(-1), m1()])]),
        times(vec![pow(n_e(), 2), pi()]),
      ),
      // (4*(-1)^n)/n^2
      2 => div(times(vec![scaled(4), m1()]), pow(n_e(), 2)),
      // (6*(2 - 2*(-1)^n + (-1)^n*n^2*Pi^2))/(n^4*Pi)
      3 => div(
        times(vec![scaled(6), bracket2()]),
        times(vec![pow(n_e(), 4), pi()]),
      ),
      _ => return Ok(unevaluated(args)),
    }
  };

  match &n_arg {
    // Numeric order: evaluate the closed form (n = 0 for sine emits the
    // same division-by-zero messages wolframscript shows, then echoes
    // the call unevaluated)
    Expr::Integer(n_val) => {
      let evaluated = crate::evaluator::evaluate_expr_to_expr(&general)?;
      if *n_val == 0 {
        return Ok(unevaluated(args));
      }
      Ok(evaluated)
    }
    _ => Ok(general),
  }
}
