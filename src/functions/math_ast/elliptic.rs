#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr};

/// True if `e` contains an inexact (machine) number. Elliptic functions
/// numericize only when an argument is inexact; exact (integer/rational)
/// arguments stay symbolic, matching wolframscript (`N[...]` makes the
/// arguments inexact first via the NumericFunction path).
fn expr_is_inexact(e: &Expr) -> bool {
  match e {
    Expr::Real(_) | Expr::BigFloat(_, _) => true,
    Expr::BinaryOp { left, right, .. } => {
      expr_is_inexact(left) || expr_is_inexact(right)
    }
    Expr::UnaryOp { operand, .. } => expr_is_inexact(operand),
    Expr::FunctionCall { args, .. } | Expr::List(args) => {
      args.iter().any(expr_is_inexact)
    }
    _ => false,
  }
}

/// EllipticK[m] - Complete elliptic integral of the first kind
pub fn elliptic_k_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "EllipticK expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    Expr::Integer(0) => {
      // EllipticK[0] = Pi/2
      Ok(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Identifier("Pi".to_string())),
        right: Box::new(Expr::Integer(2)),
      })
    }
    Expr::Integer(1) => {
      // EllipticK[1] = ComplexInfinity (pole)
      Ok(Expr::Identifier("ComplexInfinity".to_string()))
    }
    Expr::Real(f) => {
      if *f == 1.0 {
        Ok(Expr::Identifier("ComplexInfinity".to_string()))
      } else if *f < 1.0 {
        // Compute via arithmetic-geometric mean: K(m) = pi / (2 * AGM(1, sqrt(1 - m)))
        Ok(Expr::Real(elliptic_k(*f)))
      } else {
        // m > 1 requires complex numbers, return unevaluated
        Ok(Expr::FunctionCall {
          name: "EllipticK".to_string(),
          args: args.to_vec().into(),
        })
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "EllipticK".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// Compute complete elliptic integral K(m) using the arithmetic-geometric mean
fn elliptic_k(m: f64) -> f64 {
  let mut a = 1.0;
  let mut b = (1.0 - m).sqrt();
  for _ in 0..100 {
    let a_new = (a + b) / 2.0;
    let b_new = (a * b).sqrt();
    if (a_new - b_new).abs() < 1e-16 {
      return std::f64::consts::PI / (2.0 * a_new);
    }
    a = a_new;
    b = b_new;
  }
  std::f64::consts::PI / (2.0 * a)
}

/// EllipticNomeQ[m] - Elliptic nome q(m) = exp(-Pi * K(1-m) / K(m))
pub fn elliptic_nome_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "EllipticNomeQ expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    Expr::Integer(1) => Ok(Expr::Integer(1)),
    Expr::Real(f) => {
      if *f == 0.0 {
        Ok(Expr::Real(0.0))
      } else if *f == 1.0 {
        Ok(Expr::Real(1.0))
      } else if *f > 0.0 && *f < 1.0 {
        let k_m = elliptic_k(*f);
        let k_1m = elliptic_k(1.0 - *f);
        let q = (-std::f64::consts::PI * k_1m / k_m).exp();
        Ok(Expr::Real(q))
      } else if *f < 0.0 {
        // For m < 0, use the identity q(m) = -q(m/(m-1)), with
        // m' = m/(m-1) ∈ (0, 1).
        let m_prime = *f / (*f - 1.0);
        let k_m = elliptic_k(m_prime);
        let k_1m = elliptic_k(1.0 - m_prime);
        let q = -(-std::f64::consts::PI * k_1m / k_m).exp();
        Ok(Expr::Real(q))
      } else {
        // m > 1: complex result, return unevaluated for now
        Ok(Expr::FunctionCall {
          name: "EllipticNomeQ".to_string(),
          args: args.to_vec().into(),
        })
      }
    }
    // Symbolic special case: EllipticNomeQ[1/2] = E^(-Pi) (since K(1/2)=K(1-1/2)=K(1/2))
    Expr::FunctionCall { name, args: rargs }
      if name == "Rational" && rargs.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&rargs[0], &rargs[1]) {
        if *n == 1 && *d == 2 {
          // q(1/2) = exp(-Pi * K(1/2) / K(1/2)) = exp(-Pi)
          return Ok(Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![
              Expr::Constant("E".to_string()),
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![Expr::Integer(-1), Expr::Constant("Pi".to_string())]
                  .into(),
              },
            ]
            .into(),
          });
        }
        // For other rationals, compute numerically
        let f = *n as f64 / *d as f64;
        if f > 0.0 && f < 1.0 {
          let k_m = elliptic_k(f);
          let k_1m = elliptic_k(1.0 - f);
          let q = (-std::f64::consts::PI * k_1m / k_m).exp();
          return Ok(Expr::Real(q));
        }
      }
      Ok(Expr::FunctionCall {
        name: "EllipticNomeQ".to_string(),
        args: args.to_vec().into(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "EllipticNomeQ".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// Numeric inverse of the elliptic nome: the parameter m in [0, 1] with
/// EllipticNomeQ[m] == q. The nome q(m) = exp(-Pi K(1-m)/K(m)) increases
/// monotonically from 0 (m = 0) to 1 (m = 1), so the root is found by bisection.
fn inverse_elliptic_nome_q_numeric(q: f64) -> f64 {
  if q <= 0.0 {
    return 0.0;
  }
  if q >= 1.0 {
    return 1.0;
  }
  let nome = |m: f64| -> f64 {
    if m <= 0.0 {
      return 0.0;
    }
    if m >= 1.0 {
      return 1.0;
    }
    (-std::f64::consts::PI * elliptic_k(1.0 - m) / elliptic_k(m)).exp()
  };
  let mut lo = 0.0_f64;
  let mut hi = 1.0_f64;
  for _ in 0..200 {
    let mid = 0.5 * (lo + hi);
    if nome(mid) < q {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  0.5 * (lo + hi)
}

/// InverseEllipticNomeQ[q] - inverse elliptic nome: the parameter m with
/// EllipticNomeQ[m] == q. Exact (integer/rational) arguments stay symbolic
/// except the elementary q = 0 -> 0 and q = 1 -> 1; inexact (machine-real)
/// arguments evaluate numerically by bisection.
pub fn inverse_elliptic_nome_q_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "InverseEllipticNomeQ expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    Expr::Integer(1) => Ok(Expr::Integer(1)),
    Expr::Real(f) => Ok(Expr::Real(inverse_elliptic_nome_q_numeric(*f))),
    other => {
      if expr_is_inexact(other)
        && let Some(q) = crate::functions::math_ast::expr_to_f64(other)
      {
        return Ok(Expr::Real(inverse_elliptic_nome_q_numeric(q)));
      }
      Ok(Expr::FunctionCall {
        name: "InverseEllipticNomeQ".to_string(),
        args: args.to_vec().into(),
      })
    }
  }
}

/// EllipticE[m] - Complete elliptic integral of the second kind
pub fn elliptic_e_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "EllipticE expects 1 or 2 arguments".into(),
    ));
  }

  // Two-argument form: EllipticE[phi, m] - incomplete elliptic integral of the second kind
  if args.len() == 2 {
    let phi_expr = &args[0];
    let m_expr = &args[1];

    // EllipticE[0, m] = 0
    if is_expr_zero(phi_expr) {
      if matches!(phi_expr, Expr::Real(_)) || matches!(m_expr, Expr::Real(_)) {
        return Ok(Expr::Real(0.0));
      }
      return Ok(Expr::Integer(0));
    }

    // EllipticE[phi, 0] = phi (the integrand collapses to 1).
    if let Some(result) = elliptic_param_zero_reduces_to_first(phi_expr, m_expr)
    {
      return Ok(result);
    }

    // Numeric evaluation
    let phi_val = expr_to_f64(phi_expr);
    let m_val = expr_to_f64(m_expr);
    let is_numeric_eval = phi_val.is_some()
      && m_val.is_some()
      && (matches!(phi_expr, Expr::Real(_)) || matches!(m_expr, Expr::Real(_)));

    if is_numeric_eval {
      let phi = phi_val.unwrap();
      let m = m_val.unwrap();
      return Ok(Expr::Real(elliptic_e_incomplete(phi, m)));
    }

    return Ok(Expr::FunctionCall {
      name: "EllipticE".to_string(),
      args: args.to_vec().into(),
    });
  }

  // One-argument form: EllipticE[m] - complete elliptic integral
  match &args[0] {
    Expr::Integer(0) => {
      // EllipticE[0] = Pi/2
      Ok(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Identifier("Pi".to_string())),
        right: Box::new(Expr::Integer(2)),
      })
    }
    Expr::Integer(1) => {
      // EllipticE[1] = 1
      Ok(Expr::Integer(1))
    }
    Expr::Real(f) => {
      if *f == 0.0 {
        Ok(Expr::Real(std::f64::consts::FRAC_PI_2))
      } else if *f == 1.0 {
        Ok(Expr::Real(1.0))
      } else if *f < 1.0 {
        Ok(Expr::Real(elliptic_e(*f)))
      } else {
        Ok(Expr::FunctionCall {
          name: "EllipticE".to_string(),
          args: args.to_vec().into(),
        })
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "EllipticE".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// Compute complete elliptic integral E(m) via series expansion
/// E(m) = (pi/2) * [1 - sum_{n=1}^inf ((2n-1)!!/(2n)!!)^2 * m^n / (2n-1)]
fn elliptic_e(m: f64) -> f64 {
  // E(m) = (pi/2) * Σ_{n=0}^∞ ((2n)!/(2^n n!))^2 * (-m^n)/((2n-1)*4^n)
  // Simpler: E(m) = (pi/2) * [1 - Σ_{n=1}^∞ (1/2 choose n)^2 * m^n / (2n-1)]
  // Using the series: E(m) = pi/2 * Σ t_n where
  // t_0 = 1, t_n = t_{n-1} * ((2n-1)/(2n))^2 * m * (2n-1)/(2n+1-2) ... hmm

  // Better: use the standard power series
  // E(m) = pi/2 * [1 - (1/2)^2 * m/1 - (1*3)^2/(2*4)^2 * m^2/3 - (1*3*5)^2/(2*4*6)^2 * m^3/5 - ...]
  // Term_n = ((2n-1)!!/(2n)!!)^2 * m^n / (2n-1) for n >= 1
  let mut sum = 1.0;
  let mut coeff = 1.0; // ((2n-1)!!/(2n)!!)^2

  for n in 1..500 {
    let nf = n as f64;
    // coeff_n = coeff_{n-1} * ((2n-1)/(2n))^2
    let ratio = (2.0 * nf - 1.0) / (2.0 * nf);
    coeff *= ratio * ratio;
    let term = coeff * m.powi(n) / (2.0 * nf - 1.0);
    sum -= term;
    if term.abs() < 1e-16 * sum.abs() {
      break;
    }
  }

  std::f64::consts::FRAC_PI_2 * sum
}

/// JacobiZeta[phi, m] - Jacobi zeta function
/// Z(phi, m) = E(phi, m) - (E(m)/K(m)) * F(phi, m)
pub fn jacobi_zeta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiZeta expects exactly 2 arguments".into(),
    ));
  }

  let phi_expr = &args[0];
  let m_expr = &args[1];

  // JacobiZeta[0, m] = 0
  if is_expr_zero(phi_expr) {
    if matches!(phi_expr, Expr::Real(_)) || matches!(m_expr, Expr::Real(_)) {
      return Ok(Expr::Real(0.0));
    }
    return Ok(Expr::Integer(0));
  }

  // JacobiZeta[phi, 0] = 0
  if is_expr_zero(m_expr) {
    if matches!(phi_expr, Expr::Real(_)) || matches!(m_expr, Expr::Real(_)) {
      return Ok(Expr::Real(0.0));
    }
    return Ok(Expr::Integer(0));
  }

  // Numeric evaluation
  let phi_val = expr_to_f64(phi_expr);
  let m_val = expr_to_f64(m_expr);
  let is_numeric_eval = phi_val.is_some()
    && m_val.is_some()
    && (matches!(phi_expr, Expr::Real(_)) || matches!(m_expr, Expr::Real(_)));

  if is_numeric_eval {
    let phi = phi_val.unwrap();
    let m = m_val.unwrap();
    let e_phi_m = elliptic_e_incomplete(phi, m);
    let e_m = elliptic_e(m);
    let k_m = elliptic_k(m);
    let f_phi_m = elliptic_f(phi, m);
    return Ok(Expr::Real(e_phi_m - (e_m / k_m) * f_phi_m));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiZeta".to_string(),
    args: args.to_vec().into(),
  })
}

/// Compute incomplete elliptic integral E(phi, m) via Simpson's rule
/// E(phi, m) = integral from 0 to phi of sqrt(1 - m*sin^2(theta)) dtheta
pub fn elliptic_e_incomplete(phi: f64, m: f64) -> f64 {
  if phi == 0.0 {
    return 0.0;
  }
  let n = 1000;
  let h = phi / n as f64;
  let f = |theta: f64| (1.0 - m * theta.sin().powi(2)).sqrt();
  let mut sum = f(0.0) + f(phi);
  for i in 1..n {
    let theta = i as f64 * h;
    if i % 2 == 0 {
      sum += 2.0 * f(theta);
    } else {
      sum += 4.0 * f(theta);
    }
  }
  sum * h / 3.0
}

/// EllipticF[phi, m] - Incomplete elliptic integral of the first kind
/// F(phi, m) = integral from 0 to phi of 1/sqrt(1 - m*sin^2(theta)) dtheta
pub fn elliptic_f_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "EllipticF expects exactly 2 arguments".into(),
    ));
  }

  let phi_expr = &args[0];
  let m_expr = &args[1];

  // EllipticF[0, m] = 0 (Real if any argument is Real)
  if is_expr_zero(phi_expr) {
    if matches!(phi_expr, Expr::Real(_)) || matches!(m_expr, Expr::Real(_)) {
      return Ok(Expr::Real(0.0));
    }
    return Ok(Expr::Integer(0));
  }

  // EllipticF[phi, 0] = phi (the integrand collapses to 1).
  if let Some(result) = elliptic_param_zero_reduces_to_first(phi_expr, m_expr) {
    return Ok(result);
  }

  // Numeric evaluation
  let phi_val = expr_to_f64(phi_expr);
  let m_val = expr_to_f64(m_expr);
  let is_numeric_eval = phi_val.is_some()
    && m_val.is_some()
    && (matches!(phi_expr, Expr::Real(_)) || matches!(m_expr, Expr::Real(_)));

  if is_numeric_eval {
    let phi = phi_val.unwrap();
    let m = m_val.unwrap();
    return Ok(Expr::Real(elliptic_f(phi, m)));
  }

  Ok(Expr::FunctionCall {
    name: "EllipticF".to_string(),
    args: args.to_vec().into(),
  })
}

/// At parameter m == 0, EllipticF[phi, 0] = EllipticE[phi, 0] = phi and
/// JacobiAmplitude[u, 0] = u. Returns the first argument (made Real when any
/// argument is inexact, matching wolframscript), or None when m is not zero.
pub(crate) fn elliptic_param_zero_reduces_to_first(
  first: &Expr,
  m_expr: &Expr,
) -> Option<Expr> {
  if !is_expr_zero(m_expr) {
    return None;
  }
  let inexact =
    matches!(first, Expr::Real(_)) || matches!(m_expr, Expr::Real(_));
  if inexact && let Some(v) = expr_to_f64(first) {
    return Some(Expr::Real(v));
  }
  Some(first.clone())
}

/// Compute incomplete elliptic integral F(phi, m) via Gauss-Legendre quadrature
pub fn elliptic_f(phi: f64, m: f64) -> f64 {
  if phi == 0.0 {
    return 0.0;
  }
  // For phi = pi/2, this is the complete integral K(m)
  // Use numerical integration with adaptive Simpson's rule
  let n = 1000;
  let h = phi / n as f64;
  // Simpson's 1/3 rule
  let f = |theta: f64| 1.0 / (1.0 - m * theta.sin().powi(2)).sqrt();
  let mut sum = f(0.0) + f(phi);
  for i in 1..n {
    let theta = i as f64 * h;
    if i % 2 == 0 {
      sum += 2.0 * f(theta);
    } else {
      sum += 4.0 * f(theta);
    }
  }
  sum * h / 3.0
}

/// EllipticPi[n, m] - Complete elliptic integral of the third kind
/// EllipticPi[n, phi, m] - Incomplete elliptic integral of the third kind
/// Pi(n|m) = integral from 0 to pi/2 of 1/((1-n*sin^2(theta))*sqrt(1-m*sin^2(theta))) dtheta
pub fn elliptic_pi_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 && args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "EllipticPi expects 2 or 3 arguments".into(),
    ));
  }

  if args.len() == 2 {
    // Complete: EllipticPi[n, m]
    let n_expr = &args[0];
    let m_expr = &args[1];

    // EllipticPi[0, m] = EllipticK[m]
    if is_expr_zero(n_expr) {
      return elliptic_k_ast(&[m_expr.clone()]);
    }

    let n_val = try_eval_to_f64(n_expr);
    let m_val = try_eval_to_f64(m_expr);
    let is_numeric = n_val.is_some()
      && m_val.is_some()
      && (matches!(n_expr, Expr::Real(_)) || matches!(m_expr, Expr::Real(_)));

    if is_numeric {
      let n = n_val.unwrap();
      let m = m_val.unwrap();
      return Ok(Expr::Real(elliptic_pi_f64(
        n,
        std::f64::consts::FRAC_PI_2,
        m,
      )));
    }
  } else {
    // Incomplete: EllipticPi[n, phi, m]
    let n_expr = &args[0];
    let phi_expr = &args[1];
    let m_expr = &args[2];

    // EllipticPi[n, 0, m] = 0 (Real if any argument is Real)
    if is_expr_zero(phi_expr) {
      if matches!(n_expr, Expr::Real(_))
        || matches!(phi_expr, Expr::Real(_))
        || matches!(m_expr, Expr::Real(_))
      {
        return Ok(Expr::Real(0.0));
      }
      return Ok(Expr::Integer(0));
    }

    // EllipticPi[0, phi, m] = EllipticF[phi, m] (the characteristic drops out).
    // EllipticF then handles the further m == 0 reduction to phi.
    if is_expr_zero(n_expr) {
      return elliptic_f_ast(&[phi_expr.clone(), m_expr.clone()]);
    }

    let n_val = try_eval_to_f64(n_expr);
    let phi_val = try_eval_to_f64(phi_expr);
    let m_val = try_eval_to_f64(m_expr);
    let is_numeric = n_val.is_some()
      && phi_val.is_some()
      && m_val.is_some()
      && (matches!(n_expr, Expr::Real(_))
        || matches!(phi_expr, Expr::Real(_))
        || matches!(m_expr, Expr::Real(_)));

    if is_numeric {
      let n = n_val.unwrap();
      let phi = phi_val.unwrap();
      let m = m_val.unwrap();
      return Ok(Expr::Real(elliptic_pi_f64(n, phi, m)));
    }
  }

  Ok(Expr::FunctionCall {
    name: "EllipticPi".to_string(),
    args: args.to_vec().into(),
  })
}

/// Compute incomplete elliptic integral of the third kind via Simpson's rule
fn elliptic_pi_f64(n: f64, phi: f64, m: f64) -> f64 {
  if phi == 0.0 {
    return 0.0;
  }
  let num_steps = 1000;
  let h = phi / num_steps as f64;
  let f = |theta: f64| {
    let sin2 = theta.sin().powi(2);
    1.0 / ((1.0 - n * sin2) * (1.0 - m * sin2).sqrt())
  };
  let mut sum = f(0.0) + f(phi);
  for i in 1..num_steps {
    let theta = i as f64 * h;
    if i % 2 == 0 {
      sum += 2.0 * f(theta);
    } else {
      sum += 4.0 * f(theta);
    }
  }
  sum * h / 3.0
}

/// EllipticTheta[a, z, q] - Jacobi theta function
/// a = 1,2,3,4 selects which theta function
pub fn elliptic_theta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "EllipticTheta expects exactly 3 arguments".into(),
    ));
  }

  let a_val = match &args[0] {
    Expr::Integer(n) if *n >= 1 && *n <= 4 => *n as u32,
    // N[...] can demote the integer index to a machine Real (e.g. 3.); accept
    // an integer-valued Real so the numeric path still fires.
    Expr::Real(f) if f.fract() == 0.0 && *f >= 1.0 && *f <= 4.0 => *f as u32,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "EllipticTheta".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let q = &args[2];

  // EllipticTheta[a, z, 0]: theta1=0, theta2=0, theta3=1, theta4=1
  if is_expr_zero(q) {
    return match a_val {
      1 | 2 => Ok(Expr::Integer(0)),
      3 | 4 => Ok(Expr::Integer(1)),
      _ => unreachable!(),
    };
  }

  let z = &args[1];

  // theta1 is odd in z, so theta1(0, q) = 0 exactly for any q.
  if a_val == 1 && is_expr_zero(z) {
    return Ok(Expr::Integer(0));
  }

  // Numeric evaluation only when an argument is inexact (a machine number).
  // Exact arguments (integers/rationals) stay symbolic, matching
  // wolframscript; `N[...]` forces numericization via the NumericFunction
  // path (which makes the arguments inexact first).
  if (expr_is_inexact(z) || expr_is_inexact(q))
    && let (Some(z_f), Some(q_f)) = (expr_to_f64(z), expr_to_f64(q))
  {
    return Ok(Expr::Real(elliptic_theta_numeric(a_val, z_f, q_f)));
  }

  // Unevaluated (symbolic)
  Ok(Expr::FunctionCall {
    name: "EllipticTheta".to_string(),
    args: args.to_vec().into(),
  })
}

/// Compute Jacobi theta function numerically via series expansion
fn elliptic_theta_numeric(a: u32, z: f64, q: f64) -> f64 {
  match a {
    1 => {
      // θ₁(z, q) = 2 Σ_{n=0}^∞ (-1)^n q^{(n+1/2)²} sin((2n+1)z)
      let mut sum = 0.0;
      for n in 0..100 {
        let nf = n as f64;
        let exp = (nf + 0.5) * (nf + 0.5);
        let term = q.powf(exp) * ((2.0 * nf + 1.0) * z).sin();
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        sum += sign * term;
        if term.abs() < 1e-16 {
          break;
        }
      }
      2.0 * sum
    }
    2 => {
      // θ₂(z, q) = 2 Σ_{n=0}^∞ q^{(n+1/2)²} cos((2n+1)z)
      let mut sum = 0.0;
      for n in 0..100 {
        let nf = n as f64;
        let exp = (nf + 0.5) * (nf + 0.5);
        let term = q.powf(exp) * ((2.0 * nf + 1.0) * z).cos();
        sum += term;
        if term.abs() < 1e-16 {
          break;
        }
      }
      2.0 * sum
    }
    3 => {
      // θ₃(z, q) = 1 + 2 Σ_{n=1}^∞ q^{n²} cos(2nz)
      let mut sum = 1.0;
      for n in 1..100 {
        let nf = n as f64;
        let term = q.powf(nf * nf) * (2.0 * nf * z).cos();
        sum += 2.0 * term;
        if term.abs() < 1e-16 {
          break;
        }
      }
      sum
    }
    4 => {
      // θ₄(z, q) = 1 + 2 Σ_{n=1}^∞ (-1)^n q^{n²} cos(2nz)
      let mut sum = 1.0;
      for n in 1..100 {
        let nf = n as f64;
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        let term = q.powf(nf * nf) * (2.0 * nf * z).cos();
        sum += 2.0 * sign * term;
        if term.abs() < 1e-16 {
          break;
        }
      }
      sum
    }
    _ => unreachable!(),
  }
}

/// EllipticThetaPrime[a, z, q] - derivative of Jacobi theta function w.r.t. z
pub fn elliptic_theta_prime_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "EllipticThetaPrime expects exactly 3 arguments".into(),
    ));
  }

  let a_val = match &args[0] {
    Expr::Integer(n) if *n >= 1 && *n <= 4 => *n as u32,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "EllipticThetaPrime".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let z = &args[1];
  let q = &args[2];

  if let (Some(z_f), Some(q_f)) = (expr_to_f64(z), expr_to_f64(q)) {
    return Ok(Expr::Real(elliptic_theta_prime_numeric(a_val, z_f, q_f)));
  }

  Ok(Expr::FunctionCall {
    name: "EllipticThetaPrime".to_string(),
    args: args.to_vec().into(),
  })
}

/// Compute derivative of Jacobi theta function numerically via series expansion
fn elliptic_theta_prime_numeric(a: u32, z: f64, q: f64) -> f64 {
  match a {
    1 => {
      // θ₁'(z, q) = 2 Σ_{n=0}^∞ (-1)^n q^{(n+1/2)²} (2n+1) cos((2n+1)z)
      let mut sum = 0.0;
      for n in 0..100 {
        let nf = n as f64;
        let exp = (nf + 0.5) * (nf + 0.5);
        let coeff = 2.0 * nf + 1.0;
        let term = q.powf(exp) * coeff * (coeff * z).cos();
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        sum += sign * term;
        if term.abs() < 1e-16 {
          break;
        }
      }
      2.0 * sum
    }
    2 => {
      // θ₂'(z, q) = -2 Σ_{n=0}^∞ q^{(n+1/2)²} (2n+1) sin((2n+1)z)
      let mut sum = 0.0;
      for n in 0..100 {
        let nf = n as f64;
        let exp = (nf + 0.5) * (nf + 0.5);
        let coeff = 2.0 * nf + 1.0;
        let term = q.powf(exp) * coeff * (coeff * z).sin();
        sum += term;
        if term.abs() < 1e-16 {
          break;
        }
      }
      -2.0 * sum
    }
    3 => {
      // θ₃'(z, q) = -4 Σ_{n=1}^∞ n q^{n²} sin(2nz)
      let mut sum = 0.0;
      for n in 1..100 {
        let nf = n as f64;
        let term = nf * q.powf(nf * nf) * (2.0 * nf * z).sin();
        sum += term;
        if term.abs() < 1e-16 {
          break;
        }
      }
      -4.0 * sum
    }
    4 => {
      // θ₄'(z, q) = -4 Σ_{n=1}^∞ (-1)^n n q^{n²} sin(2nz)
      let mut sum = 0.0;
      for n in 1..100 {
        let nf = n as f64;
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        let term = nf * q.powf(nf * nf) * (2.0 * nf * z).sin();
        sum += sign * term;
        if term.abs() < 1e-16 {
          break;
        }
      }
      -4.0 * sum
    }
    _ => unreachable!(),
  }
}

/// DedekindEta[τ] - Dedekind eta modular elliptic function η(τ).
///
/// Defined for τ in the upper half-plane (Im(τ) > 0) by
///   η(τ) = e^(πiτ/12) * Product[(1 - e^(2πinτ)), {n, 1, ∞}].
///
/// We also recognise the exact value η(i) = Γ(1/4)/(2*π^(3/4)).
pub fn dedekind_eta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "DedekindEta expects exactly 1 argument".into(),
    ));
  }

  // Exact case: η(I) = Gamma[1/4] / (2 * Pi^(3/4))
  if matches!(&args[0], Expr::Identifier(s) if s == "I") {
    // Construct Gamma[1/4] / (2*Pi^(3/4)) explicitly.
    let gamma_quarter = Expr::FunctionCall {
      name: "Gamma".to_string(),
      args: vec![Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(4)].into(),
      }]
      .into(),
    };
    let pi_three_quarters = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![
        Expr::Identifier("Pi".to_string()),
        Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(3), Expr::Integer(4)].into(),
        },
      ]
      .into(),
    };
    let denom = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(2)),
      right: Box::new(pi_three_quarters),
    };
    let result = Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(gamma_quarter),
      right: Box::new(denom),
    };
    return crate::evaluator::evaluate_expr_to_expr(&result);
  }

  // Numeric evaluation when τ is a complex number with Im(τ) > 0
  // AND it contains a floating-point component (otherwise stay symbolic
  // — e.g. `2*I` should not auto-evaluate numerically).
  let has_real_part = expr_contains_real(&args[0]);
  if has_real_part
    && let Some((re, im)) =
      crate::functions::math_ast::try_extract_complex_float(&args[0])
    && im > 0.0
  {
    let (eta_re, eta_im) = dedekind_eta_numeric(re, im);
    return Ok(crate::functions::math_ast::build_complex_float_expr(
      eta_re, eta_im,
    ));
  }

  Ok(Expr::FunctionCall {
    name: "DedekindEta".to_string(),
    args: args.to_vec().into(),
  })
}

fn expr_contains_real(expr: &Expr) -> bool {
  match expr {
    Expr::Real(_) | Expr::BigFloat(_, _) => true,
    Expr::BinaryOp { left, right, .. } => {
      expr_contains_real(left) || expr_contains_real(right)
    }
    Expr::UnaryOp { operand, .. } => expr_contains_real(operand),
    Expr::FunctionCall { args, .. } => args.iter().any(expr_contains_real),
    _ => false,
  }
}

/// Compute η(τ) for τ = re + im*i, im > 0.
/// Uses the q-series with q = exp(2πiτ), |q| = exp(-2π*im) < 1.
fn dedekind_eta_numeric(re: f64, im: f64) -> (f64, f64) {
  let pi = std::f64::consts::PI;

  // q = exp(2*pi*i*tau) = exp(-2*pi*im) * exp(2*pi*i*re)
  let q_abs = (-2.0 * pi * im).exp();
  let q_arg = 2.0 * pi * re;
  let (q_re, q_im) = (q_abs * q_arg.cos(), q_abs * q_arg.sin());

  // Prefactor exp(pi*i*tau/12) = exp(-pi*im/12) * exp(pi*i*re/12)
  let pre_abs = (-pi * im / 12.0).exp();
  let pre_arg = pi * re / 12.0;
  let (pre_re, pre_im) = (pre_abs * pre_arg.cos(), pre_abs * pre_arg.sin());

  // Product over n = 1 .. of (1 - q^n).
  // q^n recurrence: q^(n+1) = q^n * q (complex multiplication).
  let mut prod_re = 1.0;
  let mut prod_im = 0.0;
  let mut qn_re = q_re;
  let mut qn_im = q_im;
  for _ in 1..500 {
    // (1 - q^n)
    let factor_re = 1.0 - qn_re;
    let factor_im = -qn_im;
    // prod *= factor
    let new_re = prod_re * factor_re - prod_im * factor_im;
    let new_im = prod_re * factor_im + prod_im * factor_re;
    let delta = (new_re - prod_re).hypot(new_im - prod_im);
    prod_re = new_re;
    prod_im = new_im;
    if delta < 1e-17 * prod_re.hypot(prod_im).max(1e-300) {
      break;
    }
    // q^(n+1) = q^n * q
    let nqn_re = qn_re * q_re - qn_im * q_im;
    let nqn_im = qn_re * q_im + qn_im * q_re;
    qn_re = nqn_re;
    qn_im = nqn_im;
  }

  // result = pre * prod
  let res_re = pre_re * prod_re - pre_im * prod_im;
  let res_im = pre_re * prod_im + pre_im * prod_re;
  (res_re, res_im)
}

// ─── ModularLambda / KleinInvariantJ ───────────────────────────────────

type C = (f64, f64);
fn cmul(a: C, b: C) -> C {
  (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn cdiv(a: C, b: C) -> C {
  let d = b.0 * b.0 + b.1 * b.1;
  ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}
fn cadd(a: C, b: C) -> C {
  (a.0 + b.0, a.1 + b.1)
}
fn csub(a: C, b: C) -> C {
  (a.0 - b.0, a.1 - b.1)
}
fn cscale(a: C, s: f64) -> C {
  (a.0 * s, a.1 * s)
}
fn cpowi(a: C, n: u32) -> C {
  let mut r = (1.0, 0.0);
  for _ in 0..n {
    r = cmul(r, a);
  }
  r
}

/// ModularLambda(τ) = (θ₂(0,q)/θ₃(0,q))^4 with nome q = e^(iπτ), for
/// τ = re + im·i, im > 0. Returned as a complex (re, im).
///     θ₂(0,q) = 2 Σ_{n≥0} q^{(n+1/2)²},  θ₃(0,q) = 1 + 2 Σ_{n≥1} q^{n²}.
/// Since q = r·e^{iφ} with r = e^{-π·im}, φ = π·re, the exponents are real
/// and q^e = r^e·(cos(eφ) + i·sin(eφ)).
fn modular_lambda_numeric(re: f64, im: f64) -> C {
  let pi = std::f64::consts::PI;
  let r = (-pi * im).exp();
  let phi = pi * re;
  let mut t2: C = (0.0, 0.0);
  let mut t3: C = (1.0, 0.0);
  for n in 0..5000u32 {
    let e2 = (n as f64 + 0.5).powi(2);
    let mag2 = r.powf(e2);
    t2 = cadd(t2, (mag2 * (e2 * phi).cos(), mag2 * (e2 * phi).sin()));
    let mut mag3 = 0.0;
    if n >= 1 {
      let e3 = (n as f64).powi(2);
      mag3 = r.powf(e3);
      t3 = cadd(
        t3,
        (2.0 * mag3 * (e3 * phi).cos(), 2.0 * mag3 * (e3 * phi).sin()),
      );
    }
    if mag2 < 1e-18 && (n == 0 || mag3 < 1e-18) {
      break;
    }
  }
  t2 = cscale(t2, 2.0);
  cpowi(cdiv(t2, t3), 4)
}

/// Jacobi theta constants (θ₂, θ₃, θ₄)(0, q) for a nome q = r·e^{iφ}, |q| < 1.
///     θ₂ = 2 Σ_{n≥0} q^{(n+1/2)²},  θ₃ = 1 + 2 Σ_{n≥1} q^{n²},
///     θ₄ = 1 + 2 Σ_{n≥1} (-1)ⁿ q^{n²}.
fn theta_constants(r: f64, phi: f64) -> (C, C, C) {
  let mut t2: C = (0.0, 0.0);
  let mut t3: C = (1.0, 0.0);
  let mut t4: C = (1.0, 0.0);
  for n in 0..5000u32 {
    let e2 = (n as f64 + 0.5).powi(2);
    let mag2 = r.powf(e2);
    t2 = cadd(t2, (mag2 * (e2 * phi).cos(), mag2 * (e2 * phi).sin()));
    let mut mag3 = 0.0;
    if n >= 1 {
      let e3 = (n as f64).powi(2);
      mag3 = r.powf(e3);
      let term = (2.0 * mag3 * (e3 * phi).cos(), 2.0 * mag3 * (e3 * phi).sin());
      t3 = cadd(t3, term);
      let sign = if n.is_multiple_of(2) { 1.0 } else { -1.0 };
      t4 = cadd(t4, cscale(term, sign));
    }
    if mag2 < 1e-18 && (n == 0 || mag3 < 1e-18) {
      break;
    }
  }
  (cscale(t2, 2.0), t3, t4)
}

/// Invariants {g₂, g₃} of the Weierstrass ℘-function for the lattice with
/// half-periods ω₁, ω₂ (complex). Returns None when the half-periods are
/// (numerically) collinear, i.e. their ratio is real. Using τ = ω₂/ω₁ with
/// Im(τ) > 0 (swap the two if not), nome q = e^{iπτ}, and the theta constants:
///     e₁ = c(θ₃⁴+θ₄⁴)/3,  e₂ = c(θ₂⁴−θ₄⁴)/3,  e₃ = −c(θ₂⁴+θ₃⁴)/3,
/// with c = (π/(2ω₁))², then g₂ = 2(e₁²+e₂²+e₃²), g₃ = 4 e₁ e₂ e₃.
fn weierstrass_invariants_numeric(w1: C, w2: C) -> Option<(C, C)> {
  let pi = std::f64::consts::PI;
  let (mut a, mut b) = (w1, w2);
  let mut tau = cdiv(b, a);
  if tau.1 < 0.0 {
    std::mem::swap(&mut a, &mut b);
    tau = cdiv(b, a);
  }
  // Collinear (real ratio) half-periods do not define a lattice.
  if tau.1.abs() <= 1e-12 * tau.0.hypot(tau.1).max(1.0) {
    return None;
  }
  let r = (-pi * tau.1).exp();
  let phi = pi * tau.0;
  let (t2, t3, t4) = theta_constants(r, phi);
  let (t2_4, t3_4, t4_4) = (cpowi(t2, 4), cpowi(t3, 4), cpowi(t4, 4));
  let c = cpowi(cdiv((pi, 0.0), (2.0 * a.0, 2.0 * a.1)), 2);
  let third = 1.0 / 3.0;
  let e1 = cmul(c, cscale(cadd(t3_4, t4_4), third));
  let e2 = cmul(c, cscale(csub(t2_4, t4_4), third));
  let e3 = cmul(c, cscale(cadd(t2_4, t3_4), -third));
  let g2 = cscale(cadd(cadd(cpowi(e1, 2), cpowi(e2, 2)), cpowi(e3, 2)), 2.0);
  let g3 = cscale(cmul(cmul(e1, e2), e3), 4.0);
  Some((g2, g3))
}

/// The two lattices with extra symmetry have closed-form invariants that
/// wolframscript returns even for *exact* half-periods. For the square
/// (lemniscatic) lattice with ω₂/ω₁ = ±I the invariants are
/// {Gamma[1/4]^8 / (256 Pi^2 ω₁^4), 0}; for the hexagonal (equianharmonic)
/// lattice with ω₂/ω₁ a primitive 6th root of unity (exp(±I π/3) or
/// exp(±2 I π/3)) they are {0, Gamma[1/3]^18 / (4096 Pi^6 ω₁^6)}.
///
/// The scale factor ω₁^4 (resp. ω₁^6) is invariant under swapping the two
/// half-periods or multiplying ω₂ by the symmetry root, so it does not matter
/// which period is taken as ω₁. Returns Ok(None) for every other lattice
/// (no exact closed form; wolframscript leaves those unevaluated too).
fn weierstrass_cm_invariants(
  w1e: &Expr,
  w2e: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  // Evaluate the (possibly transcendental) half-periods to complex floats via
  // N[…] so ratios like Exp[I π/3] or (1 + I Sqrt[3])/2 are recognised.
  let to_complex = |e: &Expr| -> Option<(f64, f64)> {
    let n = Expr::FunctionCall {
      name: "N".to_string(),
      args: vec![e.clone()].into(),
    };
    let r = crate::evaluator::evaluate_expr_to_expr(&n).ok()?;
    crate::functions::math_ast::try_extract_complex_float(&r)
  };
  let (w1, w2) = match (to_complex(w1e), to_complex(w2e)) {
    (Some(a), Some(b)) => (a, b),
    _ => return Ok(None),
  };
  // τ = ω₂/ω₁
  let den = w1.0 * w1.0 + w1.1 * w1.1;
  if den <= 0.0 {
    return Ok(None);
  }
  let tau_re = (w2.0 * w1.0 + w2.1 * w1.1) / den;
  let tau_im = (w2.1 * w1.0 - w2.0 * w1.1) / den;
  let tol = 1e-9;
  let close = |a: f64, b: f64| (a - b).abs() < tol;
  // Square lattice: τ = ±I (Re τ = 0, |Im τ| = 1).
  let is_square = close(tau_re, 0.0) && close(tau_im.abs(), 1.0);
  // Hexagonal lattice: |τ| = 1 and |Re τ| = 1/2.
  let is_hex = close(tau_re.hypot(tau_im), 1.0) && close(tau_re.abs(), 0.5);
  if !is_square && !is_hex {
    return Ok(None);
  }
  let pow = |base: Expr, exp: i128| Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(base),
    right: Box::new(Expr::Integer(exp)),
  };
  let gamma = |num: i128, den: i128| Expr::FunctionCall {
    name: "Gamma".to_string(),
    args: vec![Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(num), Expr::Integer(den)].into(),
    }]
    .into(),
  };
  let pi = Expr::Identifier("Pi".to_string());
  let build = |gamma_arg_den: i128,
               gamma_pow: i128,
               coeff: i128,
               pi_pow: i128,
               w_pow: i128| Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(pow(gamma(1, gamma_arg_den), gamma_pow)),
    right: Box::new(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::Integer(coeff),
        pow(pi.clone(), pi_pow),
        pow(w1e.clone(), w_pow),
      ]
      .into(),
    }),
  };
  let (g2, g3) = if is_square {
    // Gamma[1/4]^8 / (256 Pi^2 ω₁^4)
    (build(4, 8, 256, 2, 4), Expr::Integer(0))
  } else {
    // Gamma[1/3]^18 / (4096 Pi^6 ω₁^6)
    (Expr::Integer(0), build(3, 18, 4096, 6, 6))
  };
  let g2 = crate::evaluator::evaluate_expr_to_expr(&g2)?;
  let g3 = crate::evaluator::evaluate_expr_to_expr(&g3)?;
  Ok(Some(Expr::List(vec![g2, g3].into())))
}

/// WeierstrassInvariants[{ω₁, ω₂}] — the invariants {g₂, g₃} of the lattice
/// generated by the half-periods ω₁, ω₂. Numeric when a half-period is inexact;
/// exact half-periods stay symbolic except for the two CM lattices (square and
/// hexagonal), which have the closed forms wolframscript returns.
pub fn weierstrass_invariants_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "WeierstrassInvariants expects exactly 1 argument".into(),
    ));
  }
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "WeierstrassInvariants".to_string(),
      args: args.to_vec().into(),
    })
  };
  let periods = match &args[0] {
    Expr::List(items) if items.len() == 2 => items,
    _ => return unevaluated(),
  };
  if !expr_contains_real(&periods[0]) && !expr_contains_real(&periods[1]) {
    // Exact half-periods: only the square and hexagonal CM lattices have a
    // closed form; every other exact lattice stays symbolic.
    if let Some(cm) = weierstrass_cm_invariants(&periods[0], &periods[1])? {
      return Ok(cm);
    }
    return unevaluated();
  }
  let (w1, w2) = match (
    crate::functions::math_ast::try_extract_complex_float(&periods[0]),
    crate::functions::math_ast::try_extract_complex_float(&periods[1]),
  ) {
    (Some(a), Some(b)) => (a, b),
    _ => return unevaluated(),
  };
  match weierstrass_invariants_numeric(w1, w2) {
    Some((g2, g3)) => Ok(Expr::List(
      vec![
        crate::functions::math_ast::build_complex_float_expr_keep_real(
          g2.0, g2.1,
        ),
        crate::functions::math_ast::build_complex_float_expr_keep_real(
          g3.0, g3.1,
        ),
      ]
      .into(),
    )),
    None => unevaluated(),
  }
}

/// WeierstrassHalfPeriods[{g₂, g₃}] — the fundamental half-periods {ω₁, ω₂} of
/// the lattice with the given invariants. Handled for the real, positive-
/// discriminant regime (g₂³ − 27 g₃² > 0), where the ℘ cubic 4t³ − g₂t − g₃
/// has three real roots e₁ > e₂ > e₃ and the lattice is rectangular:
///     ω₁ = K(m)/√(e₁−e₃),  ω₂ = i·K(1−m)/√(e₁−e₃),  m = (e₂−e₃)/(e₁−e₃).
/// Numeric only when an argument is inexact (matching wolframscript). The
/// rhombic case (negative discriminant) and complex invariants stay symbolic.
pub fn weierstrass_half_periods_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "WeierstrassHalfPeriods expects exactly 1 argument".into(),
    ));
  }
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "WeierstrassHalfPeriods".to_string(),
      args: args.to_vec().into(),
    })
  };
  let items = match &args[0] {
    Expr::List(items) if items.len() == 2 => items,
    _ => return unevaluated(),
  };
  if !expr_contains_real(&items[0]) && !expr_contains_real(&items[1]) {
    return unevaluated();
  }
  // Real invariants only; a complex g₂/g₃ falls back to symbolic.
  let (g2, g3) = match (
    crate::functions::math_ast::try_extract_complex_float(&items[0]),
    crate::functions::math_ast::try_extract_complex_float(&items[1]),
  ) {
    (Some((a, ai)), Some((b, bi))) if ai == 0.0 && bi == 0.0 => (a, b),
    _ => return unevaluated(),
  };
  // Positive discriminant ⇒ three real roots ⇒ rectangular lattice (this also
  // forces g₂ > 0, so the depressed cubic is in the casus irreducibilis).
  let disc = g2 * g2 * g2 - 27.0 * g3 * g3;
  if disc <= 0.0 || g2 <= 0.0 {
    return unevaluated();
  }
  // Roots of 4t³ − g₂t − g₃ = 0 via the trigonometric method for t³ + pt + q.
  let p = -g2 / 4.0;
  let q = -g3 / 4.0;
  let pi = std::f64::consts::PI;
  let amp = 2.0 * (-p / 3.0).sqrt();
  let phi = ((3.0 * q) / (2.0 * p) * (-3.0 / p).sqrt())
    .clamp(-1.0, 1.0)
    .acos();
  let mut e: Vec<f64> = (0..3)
    .map(|k| amp * (phi / 3.0 - 2.0 * pi * (k as f64) / 3.0).cos())
    .collect();
  e.sort_by(|a, b| b.partial_cmp(a).unwrap());
  let (e1, e2, e3) = (e[0], e[1], e[2]);
  let d = e1 - e3;
  let m = (e2 - e3) / d;
  let sq = d.sqrt();
  let omega1 = elliptic_k(m) / sq;
  let omega2_im = elliptic_k(1.0 - m) / sq;
  Ok(Expr::List(
    vec![
      Expr::Real(omega1),
      crate::functions::math_ast::build_complex_float_expr_keep_real(
        0.0, omega2_im,
      ),
    ]
    .into(),
  ))
}

/// ModularLambda[τ] — the elliptic modular lambda function.
pub fn modular_lambda_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ModularLambda expects exactly 1 argument".into(),
    ));
  }
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "ModularLambda".to_string(),
      args: args.to_vec().into(),
    })
  };
  // Exact value at the lemniscatic point: λ(i) = 1/2.
  if matches!(&args[0], Expr::Identifier(s) if s == "I") {
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
    });
  }
  // Numeric (machine-precision) argument in the upper half-plane.
  if expr_contains_real(&args[0])
    && let Some((re, im)) =
      crate::functions::math_ast::try_extract_complex_float(&args[0])
    && im > 0.0
  {
    let (lr, li) = modular_lambda_numeric(re, im);
    return Ok(crate::functions::math_ast::build_complex_float_expr(lr, li));
  }
  unevaluated()
}

/// KleinInvariantJ[τ] — the normalized modular invariant J(τ) = j(τ)/1728.
/// In terms of λ = ModularLambda(τ):
///     J = (4/27)·(1 - λ + λ²)³ / (λ²·(1 - λ)²).
pub fn klein_invariant_j_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "KleinInvariantJ expects exactly 1 argument".into(),
    ));
  }
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "KleinInvariantJ".to_string(),
      args: args.to_vec().into(),
    })
  };
  // Exact value at the lemniscatic point: J(i) = 1.
  if matches!(&args[0], Expr::Identifier(s) if s == "I") {
    return Ok(Expr::Integer(1));
  }
  if expr_contains_real(&args[0])
    && let Some((re, im)) =
      crate::functions::math_ast::try_extract_complex_float(&args[0])
    && im > 0.0
  {
    let lam = modular_lambda_numeric(re, im);
    let one = (1.0, 0.0);
    // 1 - λ + λ²
    let num_base = cadd(csub(one, lam), cpowi(lam, 2));
    let num = cscale(cpowi(num_base, 3), 4.0 / 27.0);
    // λ²·(1 - λ)²
    let den = cmul(cpowi(lam, 2), cpowi(csub(one, lam), 2));
    let (jr, ji) = cdiv(num, den);
    // wolframscript reports the j-invariant as a complex machine number even on
    // the imaginary axis (e.g. `166.375 + 0.*I`), so keep the imaginary part.
    return Ok(
      crate::functions::math_ast::build_complex_float_expr_keep_real(jr, ji),
    );
  }
  unevaluated()
}

// ─── EllipticExp ───────────────────────────────────────────────────────
//
// EllipticExp[u, {a, b}] is the inverse of EllipticLog on the elliptic
// curve y² = x³ + a x² + b x. For real u and real positive-discriminant
// parameters, return {x, y} numerically.
//
// Derivation. Writing the integrand on [x, ∞] and substituting w = 1/√t:
//     F(x) := ∫_x^∞ dt / (2·√(t³ + a t² + b t))
//           = ∫_0^{1/√x} dw / √(1 + a w² + b w⁴)   (=: G(τ), τ = 1/√x)
// G is monotone, well-defined for all τ ≥ 0, and G(0) = 0. We invert it
// with Newton's method (G′(τ) = 1/√(1 + a τ² + b τ⁴)) and recover
// x = 1/τ², y = ±√(x³ + a x² + b x). EllipticLog has the sign convention
// `u < 0 ⇒ y > 0`, so `y = -sign(u)·|y|`.

pub fn elliptic_exp_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "EllipticExp".to_string(),
      args: args.to_vec().into(),
    });
  }
  let u = match expr_to_real_f64(&args[0]) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: "EllipticExp".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  let (a, b) = match &args[1] {
    Expr::List(items) if items.len() == 2 => {
      match (expr_to_real_f64(&items[0]), expr_to_real_f64(&items[1])) {
        (Some(a), Some(b)) => (a, b),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "EllipticExp".to_string(),
            args: args.to_vec().into(),
          });
        }
      }
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "EllipticExp".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  if u == 0.0 {
    return Ok(Expr::List(
      vec![
        Expr::Identifier("ComplexInfinity".to_string()),
        Expr::Identifier("ComplexInfinity".to_string()),
      ]
      .into(),
    ));
  }
  // Only numericize when an argument is inexact; exact arguments stay symbolic.
  if !(expr_is_inexact(&args[0]) || expr_is_inexact(&args[1])) {
    return Ok(Expr::FunctionCall {
      name: "EllipticExp".to_string(),
      args: args.to_vec().into(),
    });
  }
  match elliptic_exp_real(u, a, b) {
    Some((x, y)) => Ok(Expr::List(vec![Expr::Real(x), Expr::Real(y)].into())),
    None => Ok(Expr::FunctionCall {
      name: "EllipticExp".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

fn expr_to_real_f64(e: &Expr) -> Option<f64> {
  match e {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(r) => Some(*r),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      let n = expr_to_real_f64(&args[0])?;
      let d = expr_to_real_f64(&args[1])?;
      Some(n / d)
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut prod = 1.0;
      for a in args.iter() {
        prod *= expr_to_real_f64(a)?;
      }
      Some(prod)
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      let mut sum = 0.0;
      for a in args.iter() {
        sum += expr_to_real_f64(a)?;
      }
      Some(sum)
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      let base = expr_to_real_f64(&args[0])?;
      let exp = expr_to_real_f64(&args[1])?;
      Some(base.powf(exp))
    }
    _ => None,
  }
}

/// Compute EllipticExp[u, {a, b}] for real u, a, b. Returns None if the
/// inversion fails (e.g. u out of the real range of EllipticLog).
fn elliptic_exp_real(u: f64, a: f64, b: f64) -> Option<(f64, f64)> {
  let target = u.abs();
  // Newton's method to find τ such that G(τ) = target.
  // For small target, G(τ) ≈ τ, so τ ≈ target is a good initial guess.
  let mut tau = target;
  for _ in 0..200 {
    let val = elliptic_g(tau, a, b);
    let deriv = 1.0 / (1.0 + a * tau * tau + b * tau.powi(4)).sqrt();
    let step = (target - val) / deriv;
    let mut new_tau = tau + step;
    if !new_tau.is_finite() || new_tau <= 0.0 {
      new_tau = tau * 0.5;
    }
    if (new_tau - tau).abs() < 1e-15 * (1.0 + tau.abs()) {
      tau = new_tau;
      break;
    }
    tau = new_tau;
  }
  if !tau.is_finite() || tau <= 0.0 {
    return None;
  }
  let x = 1.0 / (tau * tau);
  let radicand = x * x * x + a * x * x + b * x;
  if radicand < 0.0 {
    return None;
  }
  let y_abs = radicand.sqrt();
  let y = if u > 0.0 { -y_abs } else { y_abs };
  Some((x, y))
}

/// G(τ) = ∫_0^τ dw / √(1 + a w² + b w⁴), via 64-point Gauss-Legendre
/// quadrature on [0, τ]. The integrand is smooth (no singularities), so
/// the rule converges to near machine precision for τ of moderate size.
fn elliptic_g(tau: f64, a: f64, b: f64) -> f64 {
  if tau <= 0.0 {
    return 0.0;
  }
  let nodes = elliptic_gl_nodes_weights();
  let half = tau * 0.5;
  let mut sum = 0.0;
  for (t, w) in nodes.iter() {
    let wval = half * (t + 1.0);
    let f = 1.0 / (1.0 + a * wval * wval + b * wval.powi(4)).sqrt();
    sum += w * f;
  }
  half * sum
}

// Cache of Gauss-Legendre 64-point nodes/weights on [-1, 1]. Computed
// once per process using the same Newton iteration as elsewhere in
// math_ast/misc_special.rs but kept local so this module does not
// depend on misc_special's internals.
fn elliptic_gl_nodes_weights() -> &'static [(f64, f64)] {
  use std::sync::OnceLock;
  static NODES: OnceLock<Vec<(f64, f64)>> = OnceLock::new();
  NODES.get_or_init(compute_gl64)
}

fn compute_gl64() -> Vec<(f64, f64)> {
  let n = 64;
  let mut out = Vec::with_capacity(n);
  for i in 0..n {
    let mut x =
      -(std::f64::consts::PI * (4 * i + 3) as f64 / (4 * n + 2) as f64).cos();
    for _ in 0..100 {
      let (p, dp) = legendre_p_and_deriv(n, x);
      let dx = -p / dp;
      x += dx;
      if dx.abs() < 1e-16 {
        break;
      }
    }
    let (_, dp) = legendre_p_and_deriv(n, x);
    let w = 2.0 / ((1.0 - x * x) * dp * dp);
    out.push((x, w));
  }
  out
}

fn legendre_p_and_deriv(n: usize, x: f64) -> (f64, f64) {
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

/// Jacobi theta series θ1..θ4 at (v, q), plus θ1'(0, q).
fn jacobi_theta1(v: f64, q: f64) -> f64 {
  let mut sum = 0.0;
  for n in 0..40 {
    let e = (n as f64) + 0.5;
    let term = q.powf(e * e) * ((2 * n + 1) as f64 * v).sin();
    sum += if n % 2 == 0 { term } else { -term };
  }
  2.0 * sum
}

fn jacobi_theta2(v: f64, q: f64) -> f64 {
  let mut sum = 0.0;
  for n in 0..40 {
    let e = (n as f64) + 0.5;
    sum += q.powf(e * e) * ((2 * n + 1) as f64 * v).cos();
  }
  2.0 * sum
}

fn jacobi_theta3(v: f64, q: f64) -> f64 {
  let mut sum = 0.0;
  for n in 1..40 {
    sum += q.powf((n * n) as f64) * (2.0 * n as f64 * v).cos();
  }
  1.0 + 2.0 * sum
}

fn jacobi_theta4(v: f64, q: f64) -> f64 {
  let mut sum = 0.0;
  for n in 1..40 {
    let term = q.powf((n * n) as f64) * (2.0 * n as f64 * v).cos();
    sum += if n % 2 == 0 { term } else { -term };
  }
  1.0 + 2.0 * sum
}

fn jacobi_theta1_prime0(q: f64) -> f64 {
  let mut sum = 0.0;
  for n in 0..40 {
    let e = (n as f64) + 0.5;
    let term = (2 * n + 1) as f64 * q.powf(e * e);
    sum += if n % 2 == 0 { term } else { -term };
  }
  2.0 * sum
}

/// Numeric Neville theta θs/θc/θd/θn at real z, m ∈ [0, 1].
fn neville_theta_numeric(kind: char, z: f64, m: f64) -> Option<f64> {
  if !(0.0..=1.0).contains(&m) || !z.is_finite() {
    return None;
  }
  if m == 0.0 {
    return Some(match kind {
      's' => z.sin(),
      'c' => z.cos(),
      _ => 1.0,
    });
  }
  if m == 1.0 {
    return Some(match kind {
      's' => z.sinh(),
      'n' => z.cosh(),
      _ => 1.0,
    });
  }
  let k = elliptic_k(m);
  let kp = elliptic_k(1.0 - m);
  let q = (-std::f64::consts::PI * kp / k).exp();
  let v = std::f64::consts::PI * z / (2.0 * k);
  Some(match kind {
    's' => {
      2.0 * k * jacobi_theta1(v, q)
        / (std::f64::consts::PI * jacobi_theta1_prime0(q))
    }
    'c' => jacobi_theta2(v, q) / jacobi_theta2(0.0, q),
    'd' => jacobi_theta3(v, q) / jacobi_theta3(0.0, q),
    _ => jacobi_theta4(v, q) / jacobi_theta4(0.0, q),
  })
}

/// NevilleThetaS/C/D/N[z, m]: exact special values at z == 0, m == 0,
/// m == 1; odd/even parity extraction on literal negative arguments;
/// machine evaluation for real z and m in [0, 1]; everything else stays
/// unevaluated (matching wolframscript, which keeps exact arguments
/// symbolic).
pub fn neville_theta_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let kind = name.as_bytes()[12].to_ascii_lowercase() as char;
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() != 2 {
    let word = if args.len() == 1 {
      "argument"
    } else {
      "arguments"
    };
    crate::emit_message(&format!(
      "{}::argr: {} called with {} {}; 2 arguments are expected.",
      name,
      name,
      args.len(),
      word
    ));
    return unevaluated();
  }
  let (z, m) = (&args[0], &args[1]);
  let trig = |head: &str, arg: Expr| {
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: head.to_string(),
      args: vec![arg].into(),
    })
  };
  // z == 0: {s, c, d, n} → {0, 1, 1, 1}.
  if matches!(z, Expr::Integer(0)) {
    return Ok(Expr::Integer(if kind == 's' { 0 } else { 1 }));
  }
  // m == 0: {Sin[z], Cos[z], 1, 1}; m == 1: {Sinh[z], 1, 1, Cosh[z]}.
  if matches!(m, Expr::Integer(0)) {
    return match kind {
      's' => trig("Sin", z.clone()),
      'c' => trig("Cos", z.clone()),
      _ => Ok(Expr::Integer(1)),
    };
  }
  if matches!(m, Expr::Integer(1)) {
    return match kind {
      's' => trig("Sinh", z.clone()),
      'n' => trig("Cosh", z.clone()),
      _ => Ok(Expr::Integer(1)),
    };
  }
  // Parity: θs is odd, the others even.
  let negated = match z {
    Expr::FunctionCall { name: tn, args: ta }
      if tn == "Times"
        && ta.len() == 2
        && matches!(&ta[0], Expr::Integer(-1)) =>
    {
      Some(ta[1].clone())
    }
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => Some((**operand).clone()),
    _ => None,
  };
  if let Some(pos) = negated {
    let inner = Expr::FunctionCall {
      name: name.to_string(),
      args: vec![pos, m.clone()].into(),
    };
    let flipped = crate::evaluator::evaluate_expr_to_expr(&inner)?;
    return if kind == 's' {
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), flipped].into(),
      })
    } else {
      Ok(flipped)
    };
  }
  // Machine evaluation only when a Real is present (exact numeric
  // arguments stay symbolic in wolframscript).
  let has_real = matches!(z, Expr::Real(_)) || matches!(m, Expr::Real(_));
  if has_real
    && let (Some(zv), Some(mv)) = (
      crate::functions::math_ast::try_eval_to_f64(z),
      crate::functions::math_ast::try_eval_to_f64(m),
    )
    && let Some(v) = neville_theta_numeric(kind, zv, mv)
  {
    return Ok(Expr::Real(v));
  }
  unevaluated()
}
