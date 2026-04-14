#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr};

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
          args: args.to_vec(),
        })
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "EllipticK".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Compute complete elliptic integral K(m) using the arithmetic-geometric mean
pub fn elliptic_k(m: f64) -> f64 {
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
      } else {
        // Outside [0, 1], return unevaluated
        Ok(Expr::FunctionCall {
          name: "EllipticNomeQ".to_string(),
          args: args.to_vec(),
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
                args: vec![Expr::Integer(-1), Expr::Constant("Pi".to_string())],
              },
            ],
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
        args: args.to_vec(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "EllipticNomeQ".to_string(),
      args: args.to_vec(),
    }),
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
      args: args.to_vec(),
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
          args: args.to_vec(),
        })
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "EllipticE".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Compute complete elliptic integral E(m) via series expansion
/// E(m) = (pi/2) * [1 - sum_{n=1}^inf ((2n-1)!!/(2n)!!)^2 * m^n / (2n-1)]
pub fn elliptic_e(m: f64) -> f64 {
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
    args: args.to_vec(),
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
    args: args.to_vec(),
  })
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
    args: args.to_vec(),
  })
}

/// Compute incomplete elliptic integral of the third kind via Simpson's rule
pub fn elliptic_pi_f64(n: f64, phi: f64, m: f64) -> f64 {
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
    _ => {
      return Ok(Expr::FunctionCall {
        name: "EllipticTheta".to_string(),
        args: args.to_vec(),
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

  // Numeric evaluation when both z and q are numeric
  let z = &args[1];
  if let (Some(z_f), Some(q_f)) = (expr_to_f64(z), expr_to_f64(q)) {
    return Ok(Expr::Real(elliptic_theta_numeric(a_val, z_f, q_f)));
  }

  // Unevaluated
  Ok(Expr::FunctionCall {
    name: "EllipticTheta".to_string(),
    args: args.to_vec(),
  })
}

/// Compute Jacobi theta function numerically via series expansion
pub fn elliptic_theta_numeric(a: u32, z: f64, q: f64) -> f64 {
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
        args: args.to_vec(),
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
    args: args.to_vec(),
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
