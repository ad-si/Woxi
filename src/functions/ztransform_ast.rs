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
use crate::syntax::Expr;

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
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      collect_factors(left, n_var, inverted, parts)
        && collect_factors(right, n_var, inverted, parts)
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
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
            op: crate::syntax::UnaryOperator::Minus,
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
      op: crate::syntax::BinaryOperator::Power,
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
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(num),
    right: Box::new(den),
  }
}

fn power(base: Expr, exp: i128) -> Expr {
  Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Power,
    left: Box::new(base),
    right: Box::new(Expr::Integer(exp)),
  }
}

fn negate(e: Expr) -> Expr {
  Expr::UnaryOp {
    op: crate::syntax::UnaryOperator::Minus,
    operand: Box::new(e),
  }
}
