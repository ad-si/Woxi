#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr};

/// Check if an expression contains any float-valued components (Real or BigFloat).
fn contains_float(expr: &Expr) -> bool {
  match expr {
    Expr::Real(_) | Expr::BigFloat(_, _) => true,
    Expr::BinaryOp { left, right, .. } => {
      contains_float(left) || contains_float(right)
    }
    Expr::UnaryOp { operand, .. } => contains_float(operand),
    Expr::FunctionCall { args, .. } => args.iter().any(contains_float),
    Expr::List(items) => items.iter().any(contains_float),
    _ => false,
  }
}

/// Zeta[s] - Riemann zeta function
/// Zeta[s, a] - Hurwitz zeta function
pub fn zeta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 2 {
    return hurwitz_zeta_ast(&args[0], &args[1], args);
  }

  // Zeta[ZetaZero[k]] = 0 (by definition of the non-trivial zeros), but only
  // when k is a concrete positive integer. For symbolic k, Wolfram leaves
  // Zeta[ZetaZero[k]] unevaluated. Zeta[ZetaZero[k, t]] = 0 still holds even
  // for symbolic k because the second argument fixes a unique zero.
  if let Expr::FunctionCall {
    name,
    args: zz_args,
  } = &args[0]
    && name == "ZetaZero"
  {
    let k_is_pos_int =
      matches!(zz_args.first(), Some(Expr::Integer(n)) if *n > 0);
    if zz_args.len() >= 2 || k_is_pos_int {
      return Ok(Expr::Integer(0));
    }
  }

  // Limits: as s -> +Infinity the series collapses to its first term, so
  // Zeta[Infinity] = 1; an undirected ComplexInfinity is Indeterminate.
  // Zeta[-Infinity] stays unevaluated (matching wolframscript).
  match &args[0] {
    Expr::Identifier(s) if s == "Infinity" => return Ok(Expr::Integer(1)),
    Expr::Identifier(s) if s == "ComplexInfinity" => {
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    _ => {}
  }

  match &args[0] {
    Expr::Integer(n) => {
      let n = *n;
      if n == 1 {
        // Zeta[1] = ComplexInfinity (pole)
        return Ok(Expr::Identifier("ComplexInfinity".to_string()));
      }
      if n == 0 {
        // Zeta[0] = -1/2
        return Ok(make_rational(-1, 2));
      }
      if n > 0 && n % 2 == 0 {
        // Positive even integer: Zeta[2n] = |B_{2n}| * 2^{2n-1} * Pi^{2n} / (2n)!
        if let Some(expr) = zeta_positive_even(n as usize) {
          return Ok(expr);
        }
      }
      if n < 0 {
        let abs_n = (-n) as usize;
        if abs_n.is_multiple_of(2) {
          // Negative even integer: Zeta[-2k] = 0 (trivial zeros)
          return Ok(Expr::Integer(0));
        }
        // Negative odd integer: Zeta[-n] = (-1)^n * B_{n+1} / (n+1)
        if let Some(expr) = zeta_negative_odd(abs_n) {
          return Ok(expr);
        }
      }
      // Positive odd integer >= 3 or overflow: return unevaluated
      Ok(Expr::FunctionCall {
        name: "Zeta".to_string(),
        args: args.to_vec().into(),
      })
    }
    Expr::Real(f) => {
      // Numeric evaluation
      let result = zeta_numeric(*f);
      Ok(Expr::Real(result))
    }
    _ => {
      // Only evaluate numerically if the argument contains float components
      // (e.g., Zeta[0.5 + 3.0*I] evaluates, but Zeta[1/2 + 3*I] stays symbolic)
      if contains_float(&args[0])
        && let Some((re, im)) =
          crate::functions::math_ast::try_extract_complex_float(&args[0])
      {
        if im != 0.0 {
          let (res_re, res_im) = zeta_numeric_complex(re, im);
          return Ok(crate::functions::math_ast::build_complex_float_expr(
            res_re, res_im,
          ));
        } else {
          return Ok(Expr::Real(zeta_numeric(re)));
        }
      }
      // Symbolic argument: return unevaluated
      Ok(Expr::FunctionCall {
        name: "Zeta".to_string(),
        args: args.to_vec().into(),
      })
    }
  }
}

/// Hurwitz zeta function Zeta[s, a]
/// HurwitzZeta[s, a] — the Hurwitz zeta function Sum_{k>=0} 1/(k+a)^s.
///
/// It coincides with the two-argument `Zeta[s, a]` everywhere except when `a`
/// is a non-positive integer: there the defining sum has a pole at k = -a, so
/// HurwitzZeta diverges (ComplexInfinity) for s > 0, whereas Wolfram's
/// `Zeta[s, a]` uses an analytic continuation that stays finite. For s <= 0
/// both reduce to the same Bernoulli-polynomial value, so we delegate to the
/// existing two-argument Zeta evaluation in every non-pole case.
pub fn hurwitz_zeta_public_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "HurwitzZeta".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() != 2 {
    return unevaluated();
  }
  let s = &args[0];
  let a = &args[1];

  // Pole: a non-positive integer with s > 0 makes the term 1/(k + a)^s at
  // k = -a equal to 1/0^s = ComplexInfinity.
  if let Expr::Integer(a_int) = a
    && *a_int <= 0
    && let Some(s_val) = expr_to_f64(s)
    && s_val > 0.0
  {
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }

  // Otherwise HurwitzZeta agrees with the two-argument Zeta. Reuse it.
  let result = zeta_ast(args)?;

  // When Zeta could not simplify it returns the unevaluated head `Zeta[s, a]`;
  // re-wrap that as HurwitzZeta to preserve the head.
  if let Expr::FunctionCall { name, args: r } = &result
    && name == "Zeta"
    && r.len() == 2
  {
    return unevaluated();
  }
  Ok(result)
}

fn hurwitz_zeta_ast(
  s_expr: &Expr,
  a_expr: &Expr,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  // Zeta[s, 1] = Zeta[s]
  if matches!(a_expr, Expr::Integer(1)) {
    return zeta_ast(&[s_expr.clone()]);
  }

  // Zeta[1, a] = ComplexInfinity (pole)
  if matches!(s_expr, Expr::Integer(1)) {
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }

  // Zeta[s, 1/2] = (-1 + 2^s) * Zeta[s] for all s
  if matches!(
    a_expr,
    Expr::FunctionCall { name, args: rargs }
      if name == "Rational" && rargs.len() == 2
        && matches!(&rargs[0], Expr::Integer(1))
        && matches!(&rargs[1], Expr::Integer(2))
  ) {
    // (-1 + 2^s) * Zeta[s]
    let two_pow_s = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(Expr::Integer(2)),
      right: Box::new(s_expr.clone()),
    };
    let factor = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(Expr::Integer(-1)),
      right: Box::new(two_pow_s),
    };
    let zeta_s = Expr::FunctionCall {
      name: "Zeta".to_string(),
      args: vec![s_expr.clone()].into(),
    };
    // Evaluate the product so integer cases simplify fully
    let product = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(factor),
      right: Box::new(zeta_s),
    };
    return crate::evaluator::evaluate_expr_to_expr(&product);
  }

  // Extract a as rational p/q if possible
  let a_rational: Option<(i128, i128)> = match a_expr {
    Expr::Integer(n) => Some((*n, 1)),
    Expr::FunctionCall { name, args: rargs }
      if name == "Rational" && rargs.len() == 2 =>
    {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&rargs[0], &rargs[1]) {
        Some((*p, *q))
      } else {
        None
      }
    }
    _ => None,
  };

  match s_expr {
    Expr::Integer(s) => {
      let s = *s;

      // Zeta[0, a] = 1/2 - a
      if s == 0 {
        if let Some((p, q)) = a_rational {
          // 1/2 - p/q = (q - 2p) / (2q)
          if let Some(two_p) = 2_i128.checked_mul(p) {
            let num = q - two_p;
            let den = 2 * q;
            return Ok(make_rational(num, den));
          }
        }
        // Symbolic a: return unevaluated
        return Ok(Expr::FunctionCall {
          name: "Zeta".to_string(),
          args: args.to_vec().into(),
        });
      }

      // Zeta[-n, a] for non-negative integer n: uses Bernoulli polynomials
      // Zeta[-n, a] = -B_{n+1}(a) / (n+1)
      if s < 0 {
        if let Some((p, q)) = a_rational {
          let abs_n = (-s) as usize;
          if let Some((num, den)) =
            bernoulli_polynomial_rational(abs_n + 1, p, q)
          {
            // -B_{n+1}(a) / (n+1)
            let result_num = -num;
            let result_den = den.checked_mul((abs_n + 1) as i128);
            if let Some(result_den) = result_den {
              return Ok(make_rational(result_num, result_den));
            }
          }
        }
        return Ok(Expr::FunctionCall {
          name: "Zeta".to_string(),
          args: args.to_vec().into(),
        });
      }

      // Positive integer s >= 2 with positive integer a:
      // Zeta[s, n] = Zeta[s] - sum_{k=1}^{n-1} 1/k^s
      if s >= 2
        && let Some((a_int, 1)) = a_rational
        && a_int >= 2
      {
        let zeta_s = zeta_ast(&[Expr::Integer(s)])?;
        // Compute sum_{k=1}^{a-1} 1/k^s as exact rational
        let mut sum_num: i128 = 0;
        let mut sum_den: i128 = 1;
        for k in 1..a_int {
          // Add 1/k^s to sum
          let k_pow = k.checked_pow(s as u32);
          if let Some(k_pow) = k_pow {
            // sum_num/sum_den + 1/k_pow
            let new_num = sum_num
              .checked_mul(k_pow)
              .and_then(|v| v.checked_add(sum_den));
            let new_den = sum_den.checked_mul(k_pow);
            if let (Some(new_num), Some(new_den)) = (new_num, new_den) {
              let g = gcd(new_num.abs(), new_den.abs());
              sum_num = new_num / g;
              sum_den = new_den / g;
            } else {
              return Ok(Expr::FunctionCall {
                name: "Zeta".to_string(),
                args: args.to_vec().into(),
              });
            }
          } else {
            return Ok(Expr::FunctionCall {
              name: "Zeta".to_string(),
              args: args.to_vec().into(),
            });
          }
        }
        let sum_rational = make_rational(sum_num, sum_den);
        return Ok(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(sum_rational),
          }),
          right: Box::new(zeta_s),
        });
      }

      // If a is a float, evaluate numerically
      if let Some(a_f) = expr_to_f64(a_expr)
        && contains_float(a_expr)
      {
        return Ok(Expr::Real(hurwitz_zeta_numeric(s as f64, a_f)));
      }

      // Return unevaluated
      Ok(Expr::FunctionCall {
        name: "Zeta".to_string(),
        args: args.to_vec().into(),
      })
    }
    Expr::Real(s_f) => {
      // Numeric evaluation
      let s = *s_f;
      match a_expr {
        Expr::Real(a) => {
          let result = hurwitz_zeta_numeric(s, *a);
          Ok(Expr::Real(result))
        }
        Expr::Integer(a) => {
          let result = hurwitz_zeta_numeric(s, *a as f64);
          Ok(Expr::Real(result))
        }
        Expr::FunctionCall { name, args: rargs }
          if name == "Rational" && rargs.len() == 2 =>
        {
          if let (Expr::Integer(p), Expr::Integer(q)) = (&rargs[0], &rargs[1]) {
            let a = *p as f64 / *q as f64;
            let result = hurwitz_zeta_numeric(s, a);
            return Ok(Expr::Real(result));
          }
          Ok(Expr::FunctionCall {
            name: "Zeta".to_string(),
            args: args.to_vec().into(),
          })
        }
        _ => {
          if contains_float(a_expr)
            && let Some((re, _im)) =
              crate::functions::math_ast::try_extract_complex_float(a_expr)
          {
            let result = hurwitz_zeta_numeric(s, re);
            return Ok(Expr::Real(result));
          }
          Ok(Expr::FunctionCall {
            name: "Zeta".to_string(),
            args: args.to_vec().into(),
          })
        }
      }
    }
    _ => {
      // Check for numeric float arguments
      if (contains_float(s_expr) || contains_float(a_expr))
        && let Some(s_f) = expr_to_f64(s_expr)
        && let Some(a_f) = expr_to_f64(a_expr)
      {
        let result = hurwitz_zeta_numeric(s_f, a_f);
        return Ok(Expr::Real(result));
      }
      Ok(Expr::FunctionCall {
        name: "Zeta".to_string(),
        args: args.to_vec().into(),
      })
    }
  }
}

/// Evaluate Bernoulli polynomial B_n(x) at rational x = p/q.
/// Returns the result as (numerator, denominator) or None on overflow.
fn bernoulli_polynomial_rational(
  n: usize,
  p: i128,
  q: i128,
) -> Option<(i128, i128)> {
  // B_n(x) = sum_{k=0}^{n} C(n,k) * B_k * x^{n-k}
  let mut sum_num: i128 = 0;
  let mut sum_den: i128 = 1;

  let mut binom: i128 = 1; // C(n, 0) = 1
  for k in 0..=n {
    if k > 0 {
      binom = binom.checked_mul((n + 1 - k) as i128)? / (k as i128);
    }
    let (bk_num, bk_den) = bernoulli_number(k)?;
    if bk_num == 0 {
      continue;
    }

    // x^{n-k} = p^{n-k} / q^{n-k}
    let exp = n - k;
    let x_num = p.checked_pow(exp as u32)?;
    let x_den = q.checked_pow(exp as u32)?;

    // term = binom * B_k * x^{n-k}
    let term_num = binom.checked_mul(bk_num)?.checked_mul(x_num)?;
    let term_den = bk_den.checked_mul(x_den)?;

    // sum += term
    let new_num = sum_num
      .checked_mul(term_den)?
      .checked_add(term_num.checked_mul(sum_den)?)?;
    let new_den = sum_den.checked_mul(term_den)?;
    let g = gcd(new_num.abs(), new_den.abs());
    sum_num = new_num / g;
    sum_den = new_den / g;
  }

  // Normalize sign
  if sum_den < 0 {
    sum_num = -sum_num;
    sum_den = -sum_den;
  }
  Some((sum_num, sum_den))
}

/// Numeric evaluation of the Hurwitz zeta function ζ(s, a)
/// using Euler-Maclaurin summation.
fn hurwitz_zeta_numeric(s: f64, a: f64) -> f64 {
  if (s - 1.0).abs() < 1e-15 {
    return f64::INFINITY;
  }

  // For s < 0.5, use the series representation directly with more terms
  // since the reflection formula is more complex for Hurwitz zeta.
  // For a = 1, reduce to Riemann zeta
  if (a - 1.0).abs() < 1e-15 {
    return zeta_numeric(s);
  }

  // Euler-Maclaurin summation for Hurwitz zeta
  // ζ(s, a) = sum_{k=0}^{N-1} (k+a)^{-s} + (N+a)^{1-s}/(s-1)
  //           + (N+a)^{-s}/2 + Bernoulli corrections
  let n: usize = 30;
  let nf = n as f64;

  let mut sum = 0.0;
  for k in 0..n {
    let term = k as f64 + a;
    if term > 0.0 {
      sum += term.powf(-s);
    }
  }

  let na = nf + a;
  // Integral correction
  sum += na.powf(1.0 - s) / (s - 1.0);
  // Endpoint correction
  sum += 0.5 * na.powf(-s);

  // Bernoulli corrections
  let bof: [f64; 10] = [
    1.0 / 12.0,
    -1.0 / 720.0,
    1.0 / 30240.0,
    -1.0 / 1209600.0,
    1.0 / 47900160.0,
    -691.0 / 1307674368000.0,
    7.0 / 523069747200.0,
    -3617.0 / 10670622842880000.0,
    43867.0 / 5109094217170944000.0,
    -174611.0 / 802857662698291200000.0,
  ];

  for (p_idx, &coeff) in bof.iter().enumerate() {
    let two_p = 2 * (p_idx + 1);
    let mut rising = 1.0;
    for j in 0..(two_p - 1) {
      rising *= s + j as f64;
    }
    sum += coeff * rising * na.powf(-(s + (two_p - 1) as f64));
  }

  sum
}

/// Try to extract a float value from an expression
fn expr_to_f64(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Real(f) => Some(*f),
    Expr::Integer(n) => Some(*n as f64),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&args[0], &args[1]) {
        Some(*p as f64 / *q as f64)
      } else {
        None
      }
    }
    _ => {
      if contains_float(expr) {
        if let Some((re, _im)) =
          crate::functions::math_ast::try_extract_complex_float(expr)
        {
          Some(re)
        } else {
          None
        }
      } else {
        None
      }
    }
  }
}

/// Compute Bernoulli number B_n as (numerator, denominator).
/// Returns None if overflow occurs during computation.
pub fn bernoulli_number(n: usize) -> Option<(i128, i128)> {
  if n == 0 {
    return Some((1, 1));
  }
  if n == 1 {
    return Some((-1, 2));
  }
  if n % 2 == 1 {
    return Some((0, 1));
  }

  // Compute all even Bernoulli numbers up to B_n using the recurrence:
  // B_m = -1/(m+1) * sum_{k=0}^{m-1} C(m+1, k) * B_k
  let mut b: Vec<(i128, i128)> = vec![(0, 1); n + 1];
  b[0] = (1, 1);
  b[1] = (-1, 2);

  for m in (2..=n).step_by(2) {
    let mut sum_n: i128 = 0;
    let mut sum_d: i128 = 1;
    let mut binom: i128 = 1; // C(m+1, 0)

    for k in 0..m {
      if k > 0 {
        // C(m+1, k) = C(m+1, k-1) * (m+2-k) / k
        binom = binom.checked_mul((m + 2 - k) as i128)? / (k as i128);
      }
      let (bk_n, bk_d) = b[k];
      if bk_n == 0 {
        continue;
      }

      // Add binom * B_k to sum: sum_n/sum_d + binom*bk_n/bk_d
      let term_n = binom.checked_mul(bk_n)?;
      let term_d = bk_d;

      let new_n = sum_n
        .checked_mul(term_d)?
        .checked_add(term_n.checked_mul(sum_d)?)?;
      let new_d = sum_d.checked_mul(term_d)?;
      let g = gcd(new_n.abs(), new_d.abs());
      sum_n = new_n / g;
      sum_d = new_d / g;
    }

    // B_m = -sum / (m+1)
    let bm_n = -sum_n;
    let bm_d = sum_d.checked_mul((m + 1) as i128)?;
    let g = gcd(bm_n.abs(), bm_d.abs());
    b[m] = (bm_n / g, bm_d / g);
  }

  Some(b[n])
}

/// Compute Zeta[2n] for positive even integer 2n.
/// Returns the exact expression |B_{2n}| * 2^(2n-1) * Pi^(2n) / (2n)!
pub fn zeta_positive_even(two_n: usize) -> Option<Expr> {
  let (b_num, b_den) = bernoulli_number(two_n)?;
  if b_num == 0 {
    return None;
  }

  // Compute coefficient: |B_{2n}| * 2^(2n-1) / (2n)!
  let mut num = b_num.abs();
  let mut den = b_den.abs();

  // Multiply by 2^(2n-1)
  for _ in 0..(two_n - 1) {
    num = num.checked_mul(2)?;
    let g = gcd(num, den);
    num /= g;
    den /= g;
  }

  // Divide by (2n)!
  for k in 1..=two_n {
    den = den.checked_mul(k as i128)?;
    let g = gcd(num, den);
    num /= g;
    den /= g;
  }

  // Build: num * Pi^(2n) / den
  let pi_power = Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(Expr::Identifier("Pi".to_string())),
    right: Box::new(Expr::Integer(two_n as i128)),
  };

  if num == 1 && den == 1 {
    Some(pi_power)
  } else if num == 1 {
    Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(pi_power),
      right: Box::new(Expr::Integer(den)),
    })
  } else if den == 1 {
    Some(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(num)),
      right: Box::new(pi_power),
    })
  } else {
    Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(num)),
        right: Box::new(pi_power),
      }),
      right: Box::new(Expr::Integer(den)),
    })
  }
}

/// Compute Zeta[-n] for negative odd integer -n.
/// Returns (-1)^n * B_{n+1} / (n+1)
pub fn zeta_negative_odd(abs_n: usize) -> Option<Expr> {
  let (b_num, b_den) = bernoulli_number(abs_n + 1)?;
  // (-1)^n * B_{n+1} / (n+1)
  let sign: i128 = if abs_n.is_multiple_of(2) { 1 } else { -1 };
  let result_num = sign.checked_mul(b_num)?;
  let result_den = b_den.checked_mul((abs_n + 1) as i128)?;
  Some(make_rational(result_num, result_den))
}

/// Compute Zeta(s) numerically for real s using Euler-Maclaurin formula.
pub fn zeta_numeric(s: f64) -> f64 {
  use std::f64::consts::PI;

  if (s - 1.0).abs() < 1e-15 {
    return f64::INFINITY;
  }

  // For s < 0.5, use the reflection formula:
  // zeta(s) = 2^s * pi^(s-1) * sin(pi*s/2) * Gamma(1-s) * zeta(1-s)
  if s < 0.5 {
    return 2.0_f64.powf(s)
      * PI.powf(s - 1.0)
      * (PI * s / 2.0).sin()
      * gamma_fn(1.0 - s)
      * zeta_numeric(1.0 - s);
  }

  // For s >= 0.5, use Euler-Maclaurin summation
  let n: usize = 20;
  let nf = n as f64;

  // Direct sum: sum_{k=1}^{N-1} k^{-s}
  let mut sum = 0.0;
  for k in 1..n {
    sum += (k as f64).powf(-s);
  }

  // Integral correction: N^{1-s} / (s-1)
  sum += nf.powf(1.0 - s) / (s - 1.0);
  // Endpoint correction: N^{-s} / 2
  sum += 0.5 * nf.powf(-s);

  // Bernoulli corrections: B_{2p}/(2p)! * prod_{j=0}^{2p-2}(s+j) * N^{-(s+2p-1)}
  let bof: [f64; 10] = [
    1.0 / 12.0,                          // B_2/2!
    -1.0 / 720.0,                        // B_4/4!
    1.0 / 30240.0,                       // B_6/6!
    -1.0 / 1209600.0,                    // B_8/8!
    1.0 / 47900160.0,                    // B_10/10!
    -691.0 / 1307674368000.0,            // B_12/12!
    7.0 / 523069747200.0,                // B_14/14!
    -3617.0 / 10670622842880000.0,       // B_16/16!
    43867.0 / 5109094217170944000.0,     // B_18/18!
    -174611.0 / 802857662698291200000.0, // B_20/20!
  ];

  for (p_idx, &coeff) in bof.iter().enumerate() {
    let two_p = 2 * (p_idx + 1);
    // Rising factorial: prod_{j=0}^{2p-2} (s+j)
    let mut rising = 1.0;
    for j in 0..(two_p - 1) {
      rising *= s + j as f64;
    }
    sum += coeff * rising * nf.powf(-(s + (two_p - 1) as f64));
  }

  sum
}

// ─── Complex arithmetic helpers ──────────────────────────────────────

fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
  (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn cdiv(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
  let d = b.0 * b.0 + b.1 * b.1;
  ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}

fn cexp(z: (f64, f64)) -> (f64, f64) {
  let mag = z.0.exp();
  (mag * z.1.cos(), mag * z.1.sin())
}

fn cln(z: (f64, f64)) -> (f64, f64) {
  let r = (z.0 * z.0 + z.1 * z.1).sqrt();
  (r.ln(), z.1.atan2(z.0))
}

fn cpow(base: (f64, f64), exp: (f64, f64)) -> (f64, f64) {
  if base.0 == 0.0 && base.1 == 0.0 {
    return (0.0, 0.0);
  }
  cexp(cmul(exp, cln(base)))
}

fn csin(z: (f64, f64)) -> (f64, f64) {
  (z.0.sin() * z.1.cosh(), z.0.cos() * z.1.sinh())
}

/// Complex Gamma function using the Lanczos approximation.
pub fn gamma_complex(re: f64, im: f64) -> (f64, f64) {
  // Reflection for Re(z) < 0.5
  if re < 0.5 {
    // Gamma(z) = pi / (sin(pi*z) * Gamma(1-z))
    let sin_piz = csin((std::f64::consts::PI * re, std::f64::consts::PI * im));
    let g1z = gamma_complex(1.0 - re, -im);
    let prod = cmul(sin_piz, g1z);
    return cdiv((std::f64::consts::PI, 0.0), prod);
  }

  // Lanczos approximation with g=7, n=9
  const P: [f64; 9] = [
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

  let z = (re - 1.0, im);
  let mut x = (P[0], 0.0);
  for i in 1..P.len() {
    let denom = (z.0 + i as f64, z.1);
    let term = cdiv((P[i], 0.0), denom);
    x.0 += term.0;
    x.1 += term.1;
  }
  let t = (z.0 + 7.5, z.1);
  let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();
  let pow = cpow(t, (z.0 + 0.5, z.1));
  let exp_neg_t = cexp((-t.0, -t.1));
  let r = cmul(cmul(pow, exp_neg_t), x);
  (sqrt_2pi * r.0, sqrt_2pi * r.1)
}

/// Complex LogGamma function, matching Wolfram's analytic-continuation
/// branch (continuous away from 0 and the negative real axis; in
/// particular `LogGamma[z] − Log[Gamma[z]]` is a multiple of 2πi for z
/// off the real axis).
///
/// Strategy: forward-shift z by enough to make Re(z+n) large, then apply
/// the complex Stirling series. The recurrence
/// `LogGamma(z) = LogGamma(z+1) − Log(z)` (with principal-branch Log)
/// gives the correct continuation.
pub fn log_gamma_complex(re: f64, im: f64) -> (f64, f64) {
  use std::f64::consts::PI;
  let log_2pi = (2.0 * PI).ln();
  // Forward-shift to |z| ≳ 11 for Stirling accuracy. Skipping the shift
  // when already large preserves precision for inputs like 12+3I.
  let mut z = (re, im);
  let mut acc_log = (0.0_f64, 0.0_f64);
  while z.0 * z.0 + z.1 * z.1 < 121.0 || z.0 < 11.0 {
    let lz = cln(z);
    acc_log = (acc_log.0 + lz.0, acc_log.1 + lz.1);
    z = (z.0 + 1.0, z.1);
  }
  // Stirling: (z - 1/2)*log(z) - z + log(2π)/2 + corrections
  let lz = cln(z);
  let z_half = (z.0 - 0.5, z.1);
  let main = cmul(z_half, lz);
  let mut result = (main.0 - z.0 + 0.5 * log_2pi, main.1 - z.1);
  // Correction series in inverse powers of z
  let inv_z = cdiv((1.0, 0.0), z);
  let inv_z2 = cmul(inv_z, inv_z);
  let mut term = inv_z;
  // 1/(12 z)
  result = (result.0 + term.0 / 12.0, result.1 + term.1 / 12.0);
  term = cmul(term, inv_z2);
  // -1/(360 z^3)
  result = (result.0 - term.0 / 360.0, result.1 - term.1 / 360.0);
  term = cmul(term, inv_z2);
  // 1/(1260 z^5)
  result = (result.0 + term.0 / 1260.0, result.1 + term.1 / 1260.0);
  term = cmul(term, inv_z2);
  // -1/(1680 z^7)
  result = (result.0 - term.0 / 1680.0, result.1 - term.1 / 1680.0);
  term = cmul(term, inv_z2);
  // 1/(1188 z^9)
  result = (result.0 + term.0 / 1188.0, result.1 + term.1 / 1188.0);
  term = cmul(term, inv_z2);
  // -691/(360360 z^11)
  result = (
    result.0 - 691.0 * term.0 / 360360.0,
    result.1 - 691.0 * term.1 / 360360.0,
  );
  // Undo the recurrence shift: subtract sum of logs we accumulated.
  (result.0 - acc_log.0, result.1 - acc_log.1)
}

/// Compute Zeta(s) numerically for complex s using Euler-Maclaurin formula
/// with functional equation for Re(s) < 0.5.
pub fn zeta_numeric_complex(s_re: f64, s_im: f64) -> (f64, f64) {
  use std::f64::consts::PI;
  let s = (s_re, s_im);

  // Check for pole at s=1
  if (s_re - 1.0).abs() < 1e-15 && s_im.abs() < 1e-15 {
    return (f64::INFINITY, 0.0);
  }

  // For Re(s) < 0.5, use the reflection formula:
  // zeta(s) = 2^s * pi^(s-1) * sin(pi*s/2) * Gamma(1-s) * zeta(1-s)
  if s_re < 0.5 {
    let two_s = cpow((2.0, 0.0), s);
    let pi_s1 = cpow((PI, 0.0), (s_re - 1.0, s_im));
    let sin_ps2 = csin((PI * s_re / 2.0, PI * s_im / 2.0));
    let g = gamma_complex(1.0 - s_re, -s_im);
    let z = zeta_numeric_complex(1.0 - s_re, -s_im);
    let r = cmul(cmul(cmul(two_s, pi_s1), sin_ps2), cmul(g, z));
    return r;
  }

  // Euler-Maclaurin summation for Re(s) >= 0.5
  let n: usize = 30;
  let nf = n as f64;

  let mut sum = (0.0, 0.0);

  // Direct sum: sum_{k=1}^{N-1} k^{-s}
  for k in 1..n {
    let term = cpow((k as f64, 0.0), (-s_re, -s_im));
    sum.0 += term.0;
    sum.1 += term.1;
  }

  // Integral correction: N^{1-s} / (s-1)
  let n1s = cpow((nf, 0.0), (1.0 - s_re, -s_im));
  let int_c = cdiv(n1s, (s_re - 1.0, s_im));
  sum.0 += int_c.0;
  sum.1 += int_c.1;

  // Endpoint correction: N^{-s} / 2
  let ns = cpow((nf, 0.0), (-s_re, -s_im));
  sum.0 += 0.5 * ns.0;
  sum.1 += 0.5 * ns.1;

  // Bernoulli corrections
  let bof: [f64; 10] = [
    1.0 / 12.0,
    -1.0 / 720.0,
    1.0 / 30240.0,
    -1.0 / 1209600.0,
    1.0 / 47900160.0,
    -691.0 / 1307674368000.0,
    7.0 / 523069747200.0,
    -3617.0 / 10670622842880000.0,
    43867.0 / 5109094217170944000.0,
    -174611.0 / 802857662698291200000.0,
  ];

  for (p_idx, &coeff) in bof.iter().enumerate() {
    let two_p = 2 * (p_idx + 1);
    // Rising factorial: prod_{j=0}^{2p-2} (s+j)
    let mut rising = (1.0, 0.0);
    for j in 0..(two_p - 1) {
      rising = cmul(rising, (s_re + j as f64, s_im));
    }
    // N^(-(s + 2p-1))
    let pow = cpow((nf, 0.0), (-(s_re + (two_p - 1) as f64), -s_im));
    let term = cmul(rising, pow);
    sum.0 += coeff * term.0;
    sum.1 += coeff * term.1;
  }

  sum
}

/// PolyGamma[z] - digamma function (equivalent to PolyGamma[0, z])
/// PolyGamma[n, z] - n-th derivative of the digamma function
pub fn polygamma_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (n_val, z_expr) = match args.len() {
    1 => (0_i128, &args[0]),
    2 => match &args[0] {
      Expr::Integer(n) => (*n, &args[1]),
      Expr::Real(f) => {
        // Real n: evaluate numerically if z is also numeric
        if let Some(z) = extract_f64(z_expr_from_args(&args[1])) {
          return Ok(Expr::Real(polygamma_numeric(*f as usize, z)));
        }
        return Ok(Expr::FunctionCall {
          name: "PolyGamma".to_string(),
          args: args.to_vec().into(),
        });
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "PolyGamma".to_string(),
          args: args.to_vec().into(),
        });
      }
    },
    _ => {
      return Err(InterpreterError::EvaluationError(
        "PolyGamma expects 1 or 2 arguments".into(),
      ));
    }
  };

  if n_val < 0 {
    return Ok(Expr::FunctionCall {
      name: "PolyGamma".to_string(),
      args: args.to_vec().into(),
    });
  }
  let n = n_val as usize;

  // Check for poles: z = 0 or negative integer
  if let Expr::Integer(z) = z_expr
    && *z <= 0
  {
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }

  match z_expr {
    Expr::Integer(z) if *z > 0 => {
      let z = *z as usize;
      if n == 0 {
        // Digamma at positive integer: psi(z) = H_{z-1} - EulerGamma
        return Ok(polygamma_digamma_integer(z));
      }
      if n % 2 == 1 {
        // Odd n: exact result via Zeta (n+1 is even)
        if let Some(expr) = polygamma_odd_integer(n, z) {
          return Ok(expr);
        }
      }
      // Even n >= 2: return unevaluated (involves odd Zeta values)
      Ok(Expr::FunctionCall {
        name: "PolyGamma".to_string(),
        args: vec![Expr::Integer(n as i128), Expr::Integer(z as i128)].into(),
      })
    }
    Expr::Real(f) => Ok(Expr::Real(polygamma_numeric(n, *f))),
    _ => {
      // Symbolic: return unevaluated in 2-arg form
      Ok(Expr::FunctionCall {
        name: "PolyGamma".to_string(),
        args: if args.len() == 1 {
          vec![Expr::Integer(0), args[0].clone()].into()
        } else {
          args.to_vec().into()
        },
      })
    }
  }
}

pub fn extract_f64(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  }
}

pub fn z_expr_from_args(expr: &Expr) -> &Expr {
  expr
}

/// Build digamma at positive integer: H_{z-1} - EulerGamma
pub fn polygamma_digamma_integer(z: usize) -> Expr {
  let euler = Expr::Identifier("EulerGamma".to_string());
  if z == 1 {
    // H_0 = 0, so result is -EulerGamma
    return Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(-1)),
      right: Box::new(euler),
    };
  }
  // Compute H_{z-1} = Σ_{k=1}^{z-1} 1/k as rational
  let (h_num, h_den) = harmonic_rational(z - 1);
  let h_expr = make_rational(h_num, h_den);
  Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(h_expr),
    right: Box::new(euler),
  }
}

/// Compute H_n = 1 + 1/2 + ... + 1/n as (numerator, denominator)
pub fn harmonic_rational(n: usize) -> (i128, i128) {
  let mut num: i128 = 0;
  let mut den: i128 = 1;
  for k in 1..=n {
    // num/den + 1/k = (num*k + den) / (den*k)
    num = num * (k as i128) + den;
    den *= k as i128;
    let g = gcd(num.abs(), den.abs());
    num /= g;
    den /= g;
  }
  (num, den)
}

/// Build exact PolyGamma[n, z] for odd n >= 1 and positive integer z.
/// Returns n! * (zeta(n+1) - partial_sum)
pub fn polygamma_odd_integer(n: usize, z: usize) -> Option<Expr> {
  // Get zeta(n+1) as a symbolic expression (raw, not multiplied by n!)
  let zeta_expr = zeta_positive_even(n + 1)?;

  // Compute n!
  let mut nfact: i128 = 1;
  for i in 2..=n {
    nfact = nfact.checked_mul(i as i128)?;
  }

  if z == 1 {
    // No partial sum. Result = n! * zeta(n+1)
    // Need to multiply the coefficient of zeta by n!
    return polygamma_multiply_zeta_by_nfact(n + 1, nfact);
  }

  // Compute partial sum = Σ_{k=1}^{z-1} 1/k^{n+1}
  let (ps_num, ps_den) = partial_sum_powers(z - 1, n + 1)?;

  // Inner expression: Plus[-partial_sum, zeta(n+1)]
  let neg_ps = make_rational(-ps_num, ps_den);
  let inner = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![neg_ps, zeta_expr].into(),
  };

  if nfact == 1 {
    // n = 1: just the inner expression
    Some(inner)
  } else {
    // n >= 3: Times[n!, inner]
    Some(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(nfact)),
      right: Box::new(inner),
    })
  }
}

/// Multiply zeta(2n) coefficient by n! and build the expression
pub fn polygamma_multiply_zeta_by_nfact(
  two_n: usize,
  nfact: i128,
) -> Option<Expr> {
  let (b_num, b_den) = bernoulli_number(two_n)?;
  if b_num == 0 {
    return None;
  }

  // Same as zeta_positive_even but multiply by nfact
  let mut num = b_num.abs().checked_mul(nfact)?;
  let mut den = b_den.abs();

  // Multiply by 2^(2n-1)
  for _ in 0..(two_n - 1) {
    num = num.checked_mul(2)?;
    let g = gcd(num, den);
    num /= g;
    den /= g;
  }

  // Divide by (2n)!
  for k in 1..=two_n {
    den = den.checked_mul(k as i128)?;
    let g = gcd(num, den);
    num /= g;
    den /= g;
  }

  let pi_power = Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(Expr::Identifier("Pi".to_string())),
    right: Box::new(Expr::Integer(two_n as i128)),
  };

  if num == 1 && den == 1 {
    Some(pi_power)
  } else if num == 1 {
    Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(pi_power),
      right: Box::new(Expr::Integer(den)),
    })
  } else if den == 1 {
    Some(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(num)),
      right: Box::new(pi_power),
    })
  } else {
    Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(num)),
        right: Box::new(pi_power),
      }),
      right: Box::new(Expr::Integer(den)),
    })
  }
}

/// Compute Σ_{k=1}^{n} 1/k^power as (numerator, denominator)
pub fn partial_sum_powers(n: usize, power: usize) -> Option<(i128, i128)> {
  let mut sum_n: i128 = 0;
  let mut sum_d: i128 = 1;
  for k in 1..=n {
    let k_pow = (k as i128).checked_pow(power as u32)?;
    let new_n = sum_n.checked_mul(k_pow)?.checked_add(sum_d)?;
    let new_d = sum_d.checked_mul(k_pow)?;
    let g = gcd(new_n.abs(), new_d.abs());
    sum_n = new_n / g;
    sum_d = new_d / g;
  }
  Some((sum_n, sum_d))
}

/// Compute polygamma function numerically
pub fn polygamma_numeric(n: usize, mut z: f64) -> f64 {
  if n == 0 {
    return digamma(z);
  }

  let sign = if n.is_multiple_of(2) { -1.0 } else { 1.0 }; // (-1)^{n+1}
  let nfact = {
    let mut f = 1.0_f64;
    for i in 2..=n {
      f *= i as f64;
    }
    f
  };

  // Use recurrence to shift z to a large value
  let mut shift_sum = 0.0;
  while z < 20.0 {
    shift_sum += 1.0 / z.powi((n + 1) as i32);
    z += 1.0;
  }

  // Asymptotic expansion for ψ^(n)(z) at large z
  // ψ^(n)(z) = (-1)^{n-1} * [(n-1)!/z^n + n!/(2z^{n+1})
  //             + Σ_k B_{2k}/(2k) * prod_{j=0}^{n-1}(2k+j) / z^{n+2k}]
  let sign_asymp = if n.is_multiple_of(2) { -1.0 } else { 1.0 }; // (-1)^{n-1}
  let nm1_fact = nfact / n as f64;

  let mut asymp = nm1_fact / z.powi(n as i32);
  asymp += nfact / (2.0 * z.powi((n + 1) as i32));

  let bernoulli = [
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
    7.0 / 6.0,
  ];
  for (ki, &b2k) in bernoulli.iter().enumerate() {
    let k = ki + 1;
    let two_k = 2 * k;
    let mut prod = 1.0;
    for j in 0..n {
      prod *= (two_k + j) as f64;
    }
    asymp += b2k / (two_k as f64) * prod / z.powi((n + two_k) as i32);
  }

  asymp *= sign_asymp;
  asymp + sign * nfact * shift_sum
}

// ─── Number Theory Functions ─────────────────────────────────────

/// LerchPhi[z, s, a] - Lerch transcendent Φ(z, s, a) = Σ_{k=0}^∞ z^k / (k+a)^s
pub fn lerch_phi_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "LerchPhi expects exactly 3 arguments".into(),
    ));
  }

  let z = &args[0];
  let s = &args[1];
  let a = &args[2];

  // Special case: z = 0 → a^(-s). Numericize only when an argument is inexact
  // (a machine number); exact arguments give the exact power
  // (e.g. LerchPhi[0, 2, 3] -> 1/9, LerchPhi[0.0, 2, 3] -> 0.1111...).
  if is_expr_zero(z) {
    if (contains_float(z) || contains_float(a) || contains_float(s))
      && let (Some(af), Some(sf)) = (try_eval_to_f64(a), try_eval_to_f64(s))
    {
      return Ok(Expr::Real(af.powf(-sf)));
    }
    // Evaluate the full a^(-s) expression so the exponent Times[-1, s] folds
    // (e.g. 3^(-2) -> 1/9) instead of staying a literal Power.
    return crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(a.clone()),
      right: Box::new(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), s.clone()].into(),
      }),
    });
  }

  // LerchPhi[z, s, 1] = PolyLog[s, z] / z (for z != 0, which the z = 0 case
  // above has already handled). Delegating to PolyLog reproduces
  // wolframscript's exact closed forms — e.g. LerchPhi[1/2, 2, 1] ->
  // 2 (Pi^2/12 - Log[2]^2/2), LerchPhi[1/3, 2, 1] -> 3 PolyLog[2, 1/3],
  // LerchPhi[z, s, 1] -> PolyLog[s, z]/z — as well as its numeric values,
  // instead of always floatifying.
  if matches!(a, Expr::Integer(1)) {
    let polylog = crate::evaluator::evaluate_function_call_ast(
      "PolyLog",
      &[s.clone(), z.clone()],
    )?;
    return crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(polylog),
      right: Box::new(z.clone()),
    });
  }

  // z = 1: LerchPhi[1, s, a] == HurwitzZeta[s, a]. Delegate for exact s and a
  // so the result stays exact (e.g. LerchPhi[1, 2, 1] -> Pi^2/6) instead of
  // being floatified, matching wolframscript.
  //
  // The delegation is restricted to the regime where Woxi's HurwitzZeta closed
  // form coincides with wolframscript's LerchPhi output: s an even integer
  // (always a Pi^(2k) closed form) or a in {1, 2} (Zeta[s] / -1 + Zeta[s]).
  // For odd s with a >= 3 wolframscript keeps a generalized-Zeta form, and
  // a <= 0 / s <= 1 diverge, so those stay on the existing paths.
  let is_exact_number = |e: &Expr| -> bool {
    matches!(e, Expr::Integer(_) | Expr::BigInteger(_))
      || matches!(e, Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2)
  };
  let s_is_even_int = matches!(s, Expr::Integer(n) if *n % 2 == 0);
  let a_is_1_or_2 = matches!(a, Expr::Integer(1) | Expr::Integer(2));
  if matches!(z, Expr::Integer(1))
    && is_exact_number(s)
    && is_exact_number(a)
    && try_eval_to_f64(s).is_some_and(|v| v > 1.0)
    && try_eval_to_f64(a).is_some_and(|v| v > 0.0)
    && (s_is_even_int || a_is_1_or_2)
  {
    let hz = crate::evaluator::evaluate_function_call_ast(
      "HurwitzZeta",
      &[s.clone(), a.clone()],
    )?;
    // Only use the closed form if HurwitzZeta actually simplified; if it
    // stayed unevaluated (e.g. a half-integer a Woxi doesn't close-form), fall
    // through to the existing numeric paths rather than emit HurwitzZeta[...].
    if !matches!(&hz, Expr::FunctionCall { name, .. } if name == "HurwitzZeta")
    {
      return Ok(hz);
    }
  }

  // Numeric evaluation.
  //   |z| < 1: wolframscript keeps exact arguments symbolic (there is no
  //     elementary closed form), so numericize only when an argument is
  //     inexact (or N[...] made it so).
  //   z == 1: wolframscript returns a HurwitzZeta-type closed form; Woxi
  //     cannot always produce it (e.g. LerchPhi[1, 2, 1/4] = π² + 8 Catalan),
  //     so fall back to the numeric value even for exact arguments.
  if let (Some(zf), Some(sf), Some(af)) =
    (try_eval_to_f64(z), try_eval_to_f64(s), try_eval_to_f64(a))
  {
    let inexact = contains_float(z) || contains_float(s) || contains_float(a);
    let z_is_one = (zf - 1.0).abs() < 1e-16 && sf > 1.0;
    let in_unit_disc = zf.abs() < 1.0;
    if z_is_one || (in_unit_disc && inexact) {
      let result = lerch_phi_numeric(zf, sf, af);
      if result.is_finite() {
        return Ok(Expr::Real(result));
      }
    }
  }

  // Analytic continuation for real z > 1, integer s ≥ 1, real a.
  // The series Σ z^k/(k+a)^s diverges, but the value is well-defined via
  // PV integral plus the iπ·residue at t = ln z. Wolfram's convention
  // for the recurrence uses `|a|^(-s)` (rather than `a^(-s)`) so that
  // both even- and odd-s cases line up with their numerical output.
  if (contains_float(z) || contains_float(s) || contains_float(a))
    && let (Some(zf), Some(sf), Some(af)) =
      (try_eval_to_f64(z), try_eval_to_f64(s), try_eval_to_f64(a))
    && zf > 1.0
    && (sf - sf.round()).abs() < 1e-12
    && (1.0..=30.0).contains(&sf)
  {
    let s_int = sf.round() as i32;
    if let Some((re, im)) = lerch_phi_z_gt_1(zf, s_int, af) {
      return Ok(complex_real(re, im));
    }
  }

  // Symbolic: return unevaluated
  Ok(Expr::FunctionCall {
    name: "LerchPhi".to_string(),
    args: args.to_vec().into(),
  })
}

/// HurwitzLerchPhi[z, s, a] - the Hurwitz-Lerch transcendent
/// Sum_{k=0}^inf z^k / (k + a)^s. It coincides with LerchPhi everywhere except
/// at a non-positive integer `a`, where the singular k = -a term is included
/// and the value is ComplexInfinity (LerchPhi instead omits that term).
pub fn hurwitz_lerch_phi_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let symbolic = || {
    Ok(Expr::FunctionCall {
      name: "HurwitzLerchPhi".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() != 3 {
    return symbolic();
  }
  // A non-positive integer third argument makes the k = -a term diverge.
  if let Expr::Integer(a) = &args[2]
    && *a <= 0
  {
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }
  // Otherwise delegate to LerchPhi; keep the HurwitzLerchPhi head when LerchPhi
  // itself stays unevaluated.
  let result = lerch_phi_ast(args)?;
  if matches!(&result, Expr::FunctionCall { name, .. } if name == "LerchPhi") {
    return symbolic();
  }
  Ok(result)
}

/// Build a `Plus[Real, Times[Real, I]]` for downstream formatting.
fn complex_real(re: f64, im: f64) -> Expr {
  if im == 0.0 {
    return Expr::Real(re);
  }
  Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      Expr::Real(re),
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Real(im), Expr::Identifier("I".to_string())].into(),
      },
    ]
    .into(),
  }
}

/// LerchPhi(z, s, a) for real z > 1, integer s ≥ 1, real a (and a + k
/// non-zero for all integers k ≥ 0). Returns the (real, imaginary)
/// pair Wolfram prints.
fn lerch_phi_z_gt_1(z: f64, s: i32, a: f64) -> Option<(f64, f64)> {
  if z <= 1.0 || s < 1 {
    return None;
  }
  // Walk a up by integer steps until it lands strictly inside (0, 1].
  // Each shift contributes `|a + k|^(-s) · z^k` to the boundary sum.
  let target_a = 1.0 - (1.0 - a).fract();
  let mut shifts = 0i32;
  let mut a_shift = a;
  let mut boundary_real = 0.0_f64;
  let mut z_pow = 1.0_f64;
  while a_shift <= 0.0 {
    if a_shift.fract().abs() < 1e-12 {
      // Hit a non-positive integer pole — undefined.
      return None;
    }
    boundary_real += z_pow * a_shift.abs().powi(-s);
    a_shift += 1.0;
    z_pow *= z;
    shifts += 1;
    if shifts > 50 {
      return None;
    }
  }
  // a_shift is now > 0. Compute LerchPhi(z, s, a_shift) via PV
  // integral + iπ·residue.
  let (re_pos, im_pos) = lerch_phi_pv(z, s, a_shift)?;
  let re = boundary_real + z_pow * re_pos;
  let im = z_pow * im_pos;
  // Eat the synthesized variable so rustc doesn't complain about `target_a`.
  let _ = target_a;
  Some((re, im))
}

/// LerchPhi(z, s, a) with a > 0, z > 1 real, integer s. Returns the
/// principal-value integral as the real part and the residue
/// contribution as the imaginary part.
fn lerch_phi_pv(z: f64, s: i32, a: f64) -> Option<(f64, f64)> {
  if a <= 0.0 || z <= 1.0 || s < 1 {
    return None;
  }
  let ln_z = z.ln();
  let gamma_s = (1..s).map(|i| i as f64).product::<f64>().max(1.0);
  // The integrand has a simple pole at t = ln(z) with residue
  //   h(ln z) = (ln z)^(s−1) · e^(−a·ln z) = (ln z)^(s−1) · z^(−a).
  let h = |t: f64| t.powi(s - 1) * (-a * t).exp();
  let h_lnz = h(ln_z);
  // Subtract the singular part: g(t) = integrand(t) - h(ln z)/(t - ln z).
  // g is smooth at t = ln z (use Taylor expansion in a small window).
  let h_prime_lnz =
    (-a * ln_z).exp() * ln_z.powi(s - 2) * (((s - 1) as f64) - a * ln_z);
  let g = |t: f64| -> f64 {
    let delta = t - ln_z;
    if delta.abs() < 1e-4 {
      // 1/(1 − z e^(−t)) = 1/[(t − ln z)(1 − (t − ln z)/2 + …)]
      //                 = 1/(t − ln z) + 1/2 + (t − ln z)/12 + …
      // ⇒ integrand − h(ln z)/(t − ln z)
      //   = (h(t) − h(ln z))/(t − ln z) + h(t)·(1/2 + (t − ln z)/12 + …)
      //   → h'(ln z) + h(ln z)/2 as t → ln z.
      h_prime_lnz + h_lnz / 2.0
    } else {
      let denom = 1.0 - z * (-t).exp();
      h(t) / denom - h_lnz / delta
    }
  };
  let t_max = (50.0 + 10.0 / a).min(200.0);
  let int_g = gauss_legendre_integrate_lerch(0.0, t_max, &g);
  let re = (int_g + h_lnz * ((t_max - ln_z) / ln_z).ln()) / gamma_s;
  let im = -std::f64::consts::PI * ln_z.powi(s - 1) * z.powf(-a) / gamma_s;
  Some((re, im))
}

/// 64-point Gauss-Legendre on `[lo, hi]`, with a fixed local node table.
fn gauss_legendre_integrate_lerch<F: Fn(f64) -> f64>(
  lo: f64,
  hi: f64,
  f: &F,
) -> f64 {
  let nodes = lerch_gl_nodes();
  let half = (hi - lo) * 0.5;
  let mid = (lo + hi) * 0.5;
  let mut sum = 0.0;
  for (t, w) in nodes.iter() {
    let x = mid + half * t;
    sum += w * f(x);
  }
  half * sum
}

fn lerch_gl_nodes() -> &'static [(f64, f64)] {
  use std::sync::OnceLock;
  static NODES: OnceLock<Vec<(f64, f64)>> = OnceLock::new();
  NODES.get_or_init(|| {
    let n = 64;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
      let mut x =
        -(std::f64::consts::PI * (4 * i + 3) as f64 / (4 * n + 2) as f64).cos();
      for _ in 0..100 {
        let (p, dp) = legendre_p_dp(n, x);
        let dx = -p / dp;
        x += dx;
        if dx.abs() < 1e-16 {
          break;
        }
      }
      let (_, dp) = legendre_p_dp(n, x);
      let w = 2.0 / ((1.0 - x * x) * dp * dp);
      out.push((x, w));
    }
    out
  })
}

fn legendre_p_dp(n: usize, x: f64) -> (f64, f64) {
  let mut p0 = 1.0;
  let mut p1 = x;
  for k in 1..n {
    let pk = ((2 * k + 1) as f64 * x * p1 - k as f64 * p0) / ((k + 1) as f64);
    p0 = p1;
    p1 = pk;
  }
  let dp = (n as f64) * (x * p1 - p0) / (x * x - 1.0);
  (p1, dp)
}

/// Compute LerchPhi numerically via series: Σ z^k / (k+a)^s
pub fn lerch_phi_numeric(z: f64, s: f64, a: f64) -> f64 {
  // For z = 1, this is the Hurwitz zeta: Σ 1/(k+a)^s
  // Use Euler-Maclaurin to add tail correction for better convergence
  let n_terms = if (z - 1.0).abs() < 1e-14 { 200 } else { 1000 };
  let mut sum = 0.0;
  let mut z_pow = 1.0; // z^k
  for k in 0..n_terms {
    let denom = (k as f64 + a).powf(s);
    if denom.abs() > 1e-300 {
      let term = z_pow / denom;
      sum += term;
      if term.abs() < 1e-15 * sum.abs() && k > 5 {
        return sum;
      }
    }
    z_pow *= z;
    if z_pow.abs() < 1e-300 {
      return sum;
    }
  }

  // For z=1 (Hurwitz zeta), add Euler-Maclaurin tail to extend the partial sum:
  //   Σ_{k=N}^∞ f(k) ≈ ∫_N^∞ f + f(N)/2 - f'(N)/12 + f'''(N)/720 - …
  // with f(t) = 1/(t+a)^s. For 200-term partial sums this brings the residual
  // below 1e-13 for moderate s, matching wolframscript at machine precision.
  if (z - 1.0).abs() < 1e-14 && s > 1.0 {
    let n = n_terms as f64;
    let na = n + a;
    let tail = na.powf(1.0 - s) / (s - 1.0);
    let f_n = 1.0 / na.powf(s);
    // -f'(N)/12 = s / (12 (N+a)^(s+1))
    let em2 = s / (12.0 * na.powf(s + 1.0));
    // f'''(N)/720 = -s(s+1)(s+2) / (720 (N+a)^(s+3))
    let em3 = -s * (s + 1.0) * (s + 2.0) / (720.0 * na.powf(s + 3.0));
    sum += tail + f_n / 2.0 + em2 + em3;
  }

  sum
}

/// PrimeZetaP[s] - Prime zeta function: P(s) = sum_{p prime} 1/p^s.
/// Computed via Möbius inversion: P(s) = sum_{k=1}^∞ μ(k)/k * log(ζ(ks)).
pub fn prime_zeta_p_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "PrimeZetaP".to_string(),
      args: args.to_vec().into(),
    });
  }

  // Only evaluate numerically for Real (approximate) arguments
  let s = match &args[0] {
    Expr::Real(v) if *v > 1.0 => *v,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PrimeZetaP".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Use Möbius inversion: P(s) = sum_{k=1}^K μ(k)/k * log(ζ(ks))
  // For s > 1, the series converges very rapidly.
  let max_k = 60;
  let mut result = 0.0_f64;

  for k in 1..=max_k {
    let mu = mobius(k);
    if mu == 0 {
      continue;
    }
    let ks = (k as f64) * s;
    // Use the evaluator's Zeta function for accuracy
    let zeta_expr =
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "N".to_string(),
        args: vec![Expr::FunctionCall {
          name: "Zeta".to_string(),
          args: vec![Expr::Real(ks)].into(),
        }]
        .into(),
      });
    let zeta_val = match zeta_expr {
      Ok(ref e) => try_eval_to_f64(e).unwrap_or(1.0),
      Err(_) => 1.0,
    };
    if zeta_val <= 1.0 {
      continue;
    }
    let term = (mu as f64) / (k as f64) * zeta_val.ln();
    result += term;
    if k > 5 && term.abs() < 1e-16 {
      break;
    }
  }

  Ok(Expr::Real(result))
}

/// Compute the Möbius function μ(n).
fn mobius(n: usize) -> i32 {
  if n == 0 {
    return 0;
  }
  if n == 1 {
    return 1;
  }
  let mut m = n;
  let mut num_factors = 0;
  let mut d = 2;
  while d * d <= m {
    if m.is_multiple_of(d) {
      m /= d;
      if m.is_multiple_of(d) {
        return 0; // p^2 divides n
      }
      num_factors += 1;
    }
    d += 1;
  }
  if m > 1 {
    num_factors += 1;
  }
  if num_factors % 2 == 0 { 1 } else { -1 }
}

/// Riemann-Siegel theta function θ(t) for real t:
///   θ(t) = arg(Γ(1/4 + i t/2)) − (t/2) log(π)
/// computed as Im(LogGamma(1/4 + i t/2)) − (t/2) log(π).
fn riemann_siegel_theta_numeric(t: f64) -> f64 {
  use std::f64::consts::PI;
  let (_re, im) = log_gamma_complex(0.25, t / 2.0);
  im - (t / 2.0) * PI.ln()
}

/// High-accuracy ζ(1/2 + i t) via Euler-Maclaurin summation with the
/// number of direct terms N scaled with t. For the critical line the
/// truncation error of the Euler-Maclaurin tail decays like N^{-(2p+1/2)}
/// while the imaginary part of the argument forces N ≳ t for the direct
/// sum to dominate the Bernoulli corrections. We pick N = max(30, t) and
/// include Bernoulli terms up to B_20.
fn zeta_half_plus_it(t: f64) -> (f64, f64) {
  let s_re = 0.5_f64;
  let s_im = t;

  let n: usize = (t.abs().ceil() as usize).max(30);
  let nf = n as f64;

  let mut sum = (0.0_f64, 0.0_f64);
  for k in 1..n {
    let term = cpow((k as f64, 0.0), (-s_re, -s_im));
    sum.0 += term.0;
    sum.1 += term.1;
  }

  // Integral correction: N^{1-s} / (s-1)
  let n1s = cpow((nf, 0.0), (1.0 - s_re, -s_im));
  let int_c = cdiv(n1s, (s_re - 1.0, s_im));
  sum.0 += int_c.0;
  sum.1 += int_c.1;

  // Endpoint correction: N^{-s} / 2
  let ns = cpow((nf, 0.0), (-s_re, -s_im));
  sum.0 += 0.5 * ns.0;
  sum.1 += 0.5 * ns.1;

  let bof: [f64; 10] = [
    1.0 / 12.0,
    -1.0 / 720.0,
    1.0 / 30240.0,
    -1.0 / 1209600.0,
    1.0 / 47900160.0,
    -691.0 / 1307674368000.0,
    7.0 / 523069747200.0,
    -3617.0 / 10670622842880000.0,
    43867.0 / 5109094217170944000.0,
    -174611.0 / 802857662698291200000.0,
  ];
  for (p_idx, &coeff) in bof.iter().enumerate() {
    let two_p = 2 * (p_idx + 1);
    let mut rising = (1.0, 0.0);
    for j in 0..(two_p - 1) {
      rising = cmul(rising, (s_re + j as f64, s_im));
    }
    let pow = cpow((nf, 0.0), (-(s_re + (two_p - 1) as f64), -s_im));
    let term = cmul(rising, pow);
    sum.0 += coeff * term.0;
    sum.1 += coeff * term.1;
  }

  sum
}

/// Riemann-Siegel Z function Z(t) for real t:
///   Z(t) = e^{i θ(t)} ζ(1/2 + i t)
/// which is real-valued for real t. Z is even: Z(-t) = Z(t).
fn riemann_siegel_z_numeric(t: f64) -> f64 {
  let t = t.abs(); // Z is even
  let theta = riemann_siegel_theta_numeric(t);
  let (zre, zim) = zeta_half_plus_it(t);
  // Re(e^{i θ} ζ) = cos θ · Re(ζ) − sin θ · Im(ζ)
  theta.cos() * zre - theta.sin() * zim
}

/// RiemannSiegelZ[t] — the Riemann-Siegel Z function.
pub fn riemann_siegel_z_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "RiemannSiegelZ expects exactly 1 argument".into(),
    ));
  }

  // Only evaluate for real numeric (Real) input; keep exact/symbolic
  // arguments unevaluated to mirror wolframscript.
  if let Some(t) = match &args[0] {
    Expr::Real(f) => Some(*f),
    _ => None,
  } {
    return Ok(Expr::Real(riemann_siegel_z_numeric(t)));
  }

  Ok(Expr::FunctionCall {
    name: "RiemannSiegelZ".to_string(),
    args: args.to_vec().into(),
  })
}

/// RiemannSiegelTheta[t] — the Riemann-Siegel theta function for real t.
pub fn riemann_siegel_theta_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "RiemannSiegelTheta expects exactly 1 argument".into(),
    ));
  }

  // Only evaluate for real numeric (Real) input; keep exact/symbolic
  // arguments unevaluated to mirror wolframscript.
  if let Expr::Real(t) = &args[0] {
    let v = riemann_siegel_theta_numeric(*t);
    // Normalise negative zero to positive zero (theta(0) = 0).
    let v = if v == 0.0 { 0.0 } else { v };
    return Ok(Expr::Real(v));
  }

  Ok(Expr::FunctionCall {
    name: "RiemannSiegelTheta".to_string(),
    args: args.to_vec().into(),
  })
}

/// DirichletEta[s] — the Dirichlet eta function: (1 - 2^(1-s)) * Zeta[s]
pub fn dirichlet_eta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "DirichletEta expects exactly 1 argument".into(),
    ));
  }

  // Handle special case s=1: eta(1) = ln(2)
  if matches!(&args[0], Expr::Integer(1)) {
    return Ok(Expr::FunctionCall {
      name: "Log".to_string(),
      args: vec![Expr::Integer(2)].into(),
    });
  }

  // For numeric (Real) input, compute directly
  if let Expr::Real(f) = &args[0] {
    // Special case: s=1.0 gives ln(2)
    if *f == 1.0 {
      return Ok(Expr::Real(2.0_f64.ln()));
    }
    let eta = (1.0 - 2.0_f64.powf(1.0 - f)) * zeta_numeric(*f);
    return Ok(num_to_expr(eta));
  }

  // For integer and rational inputs, compute the factor (1 - 2^(1-s)) exactly first
  let factor = {
    let one_minus_s = crate::evaluator::evaluate_function_call_ast(
      "Subtract",
      &[Expr::Integer(1), args[0].clone()],
    )?;
    let two_pow = crate::evaluator::evaluate_function_call_ast(
      "Power",
      &[Expr::Integer(2), one_minus_s],
    )?;
    crate::evaluator::evaluate_function_call_ast(
      "Subtract",
      &[Expr::Integer(1), two_pow],
    )?
  };

  let zeta =
    crate::evaluator::evaluate_function_call_ast("Zeta", &[args[0].clone()])?;
  crate::evaluator::evaluate_function_call_ast("Times", &[factor, zeta])
}

/// DirichletLambda[s] = sum over odd n of 1/n^s = (1 - 2^(-s)) Zeta[s].
/// (This is the Dirichlet lambda function, also written lambda(s).)
pub fn dirichlet_lambda_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "DirichletLambda expects exactly 1 argument".into(),
    ));
  }

  // For numeric (Real) input, compute directly.
  if let Expr::Real(f) = &args[0] {
    let lambda = (1.0 - 2.0_f64.powf(-f)) * zeta_numeric(*f);
    return Ok(num_to_expr(lambda));
  }

  // Otherwise build ((2^s - 1) Zeta[s]) / 2^s. Written with 2^s in the
  // denominator (rather than the equivalent (1 - 2^(-s)) Zeta[s]) to match
  // wolframscript's canonical form for symbolic/fractional s, e.g.
  // DirichletLambda[x] -> ((-1 + 2^x) Zeta[x])/2^x. For integer s the Zeta
  // closed-forms collapse it to a clean value (DirichletLambda[2] -> Pi^2/8,
  // DirichletLambda[-1] -> 1/12).
  let two_pow_s = crate::evaluator::evaluate_function_call_ast(
    "Power",
    &[Expr::Integer(2), args[0].clone()],
  )?;
  let numer_factor = crate::evaluator::evaluate_function_call_ast(
    "Subtract",
    &[two_pow_s.clone(), Expr::Integer(1)],
  )?;
  let zeta =
    crate::evaluator::evaluate_function_call_ast("Zeta", &[args[0].clone()])?;
  let numer = crate::evaluator::evaluate_function_call_ast(
    "Times",
    &[numer_factor, zeta],
  )?;
  crate::evaluator::evaluate_function_call_ast("Divide", &[numer, two_pow_s])
}

/// DirichletBeta[s] — the Dirichlet beta function β(s) = Σ (-1)^n/(2n+1)^s.
/// Closed forms: odd positive integers via Euler numbers
/// (β(2k+1) = (-1)^k E_{2k} π^(2k+1) / (4^(k+1)(2k)!)), non-positive integers
/// β(n) = EulerE[-n]/2, β(2) = Catalan. Everything else (even positive
/// integers, fractional and symbolic s) uses the Hurwitz-zeta form
/// β(s) = (Zeta[s,1/4]/2^s − Zeta[s,3/4]/2^s)/2^s, matching wolframscript.
pub fn dirichlet_beta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "DirichletBeta expects exactly 1 argument".into(),
    ));
  }
  let s = &args[0];

  // Numeric (Real) argument: β(s) = 4^(-s) (ζ(s,1/4) − ζ(s,3/4)). The Hurwitz
  // zetas individually diverge at s = 1 (their difference β(1) = π/4 is finite),
  // so that point is handled directly.
  if let Expr::Real(f) = s {
    if *f == 1.0 {
      return Ok(Expr::Real(std::f64::consts::FRAC_PI_4));
    }
    let beta = 4.0_f64.powf(-f)
      * (hurwitz_zeta_numeric(*f, 0.25) - hurwitz_zeta_numeric(*f, 0.75));
    return Ok(num_to_expr(beta));
  }

  // Exact integer argument.
  if let Some(n) = crate::functions::math_ast::expr_to_i128(s) {
    // β(n) = EulerE[-n]/2 for n <= 0.
    if n <= 0 {
      let euler = crate::evaluator::evaluate_function_call_ast(
        "EulerE",
        &[Expr::Integer(-n)],
      )?;
      return crate::evaluator::evaluate_function_call_ast(
        "Divide",
        &[euler, Expr::Integer(2)],
      );
    }
    // β(2) is Catalan's constant.
    if n == 2 {
      return Ok(Expr::Identifier("Catalan".to_string()));
    }
    // Odd positive integers: β(2k+1) = (-1)^k E_{2k} π^(2k+1) / (4^(k+1)(2k)!).
    if n % 2 == 1 {
      let k = (n - 1) / 2;
      let sign = if k % 2 == 0 { 1 } else { -1 };
      let euler = crate::evaluator::evaluate_function_call_ast(
        "EulerE",
        &[Expr::Integer(2 * k)],
      )?;
      let pi_pow = crate::evaluator::evaluate_function_call_ast(
        "Power",
        &[Expr::Identifier("Pi".to_string()), Expr::Integer(n)],
      )?;
      let numer = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[Expr::Integer(sign), euler, pi_pow],
      )?;
      let four_pow = crate::evaluator::evaluate_function_call_ast(
        "Power",
        &[Expr::Integer(4), Expr::Integer(k + 1)],
      )?;
      let fact = crate::evaluator::evaluate_function_call_ast(
        "Factorial",
        &[Expr::Integer(2 * k)],
      )?;
      let denom = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[four_pow, fact],
      )?;
      return crate::evaluator::evaluate_function_call_ast(
        "Divide",
        &[numer, denom],
      );
    }
    // Even positive integers (>= 4) fall through to the Hurwitz form.
  }

  // General form: (Zeta[s,1/4]/2^s − Zeta[s,3/4]/2^s)/2^s.
  let two_pow_s = crate::evaluator::evaluate_function_call_ast(
    "Power",
    &[Expr::Integer(2), s.clone()],
  )?;
  let z1 = crate::evaluator::evaluate_function_call_ast(
    "Zeta",
    &[s.clone(), make_rational(1, 4)],
  )?;
  let z3 = crate::evaluator::evaluate_function_call_ast(
    "Zeta",
    &[s.clone(), make_rational(3, 4)],
  )?;
  let t1 = crate::evaluator::evaluate_function_call_ast(
    "Divide",
    &[z1, two_pow_s.clone()],
  )?;
  let t3 = crate::evaluator::evaluate_function_call_ast(
    "Divide",
    &[z3, two_pow_s.clone()],
  )?;
  let diff =
    crate::evaluator::evaluate_function_call_ast("Subtract", &[t1, t3])?;
  crate::evaluator::evaluate_function_call_ast("Divide", &[diff, two_pow_s])
}
