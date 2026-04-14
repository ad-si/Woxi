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
        args: args.to_vec(),
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
        args: args.to_vec(),
      })
    }
  }
}

/// Hurwitz zeta function Zeta[s, a]
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
      args: vec![s_expr.clone()],
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
          args: args.to_vec(),
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
          args: args.to_vec(),
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
                args: args.to_vec(),
              });
            }
          } else {
            return Ok(Expr::FunctionCall {
              name: "Zeta".to_string(),
              args: args.to_vec(),
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
        args: args.to_vec(),
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
            args: args.to_vec(),
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
            args: args.to_vec(),
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
        args: args.to_vec(),
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
fn gamma_complex(re: f64, im: f64) -> (f64, f64) {
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
          args: args.to_vec(),
        });
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "PolyGamma".to_string(),
          args: args.to_vec(),
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
      args: args.to_vec(),
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
        args: vec![Expr::Integer(n as i128), Expr::Integer(z as i128)],
      })
    }
    Expr::Real(f) => Ok(Expr::Real(polygamma_numeric(n, *f))),
    _ => {
      // Symbolic: return unevaluated in 2-arg form
      Ok(Expr::FunctionCall {
        name: "PolyGamma".to_string(),
        args: if args.len() == 1 {
          vec![Expr::Integer(0), args[0].clone()]
        } else {
          args.to_vec()
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
    args: vec![neg_ps, zeta_expr],
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

  // Special case: z = 0 → a^(-s)
  if is_expr_zero(z) {
    if let (Some(af), Some(sf)) = (try_eval_to_f64(a), try_eval_to_f64(s)) {
      return Ok(Expr::Real(af.powf(-sf)));
    }
    return crate::evaluator::evaluate_function_call_ast(
      "Power",
      &[
        a.clone(),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), s.clone()],
        },
      ],
    );
  }

  // Numeric evaluation
  if let (Some(zf), Some(sf), Some(af)) =
    (try_eval_to_f64(z), try_eval_to_f64(s), try_eval_to_f64(a))
    && (zf.abs() < 1.0 || ((zf - 1.0).abs() < 1e-16 && sf > 1.0))
  {
    let result = lerch_phi_numeric(zf, sf, af);
    if result.is_finite() {
      return Ok(Expr::Real(result));
    }
  }

  // Symbolic: return unevaluated
  Ok(Expr::FunctionCall {
    name: "LerchPhi".to_string(),
    args: args.to_vec(),
  })
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

  // For z=1 (Hurwitz zeta), add integral tail: ∫_{N}^∞ 1/(t+a)^s dt = (N+a)^(1-s)/(s-1)
  if (z - 1.0).abs() < 1e-14 && s > 1.0 {
    let n = n_terms as f64;
    let tail = (n + a).powf(1.0 - s) / (s - 1.0);
    // Plus first-order Euler-Maclaurin correction: f(N)/2
    let f_n = 1.0 / (n + a).powf(s);
    sum += tail + f_n / 2.0;
  }

  sum
}

/// PrimeZetaP[s] - Prime zeta function: P(s) = sum_{p prime} 1/p^s.
/// Computed via Möbius inversion: P(s) = sum_{k=1}^∞ μ(k)/k * log(ζ(ks)).
pub fn prime_zeta_p_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "PrimeZetaP".to_string(),
      args: args.to_vec(),
    });
  }

  // Only evaluate numerically for Real (approximate) arguments
  let s = match &args[0] {
    Expr::Real(v) if *v > 1.0 => *v,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PrimeZetaP".to_string(),
        args: args.to_vec(),
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
          args: vec![Expr::Real(ks)],
        }],
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
      args: vec![Expr::Integer(2)],
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
