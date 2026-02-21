#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr};

/// JacobiP[n, a, b, x] - Jacobi polynomial P_n^{(a,b)}(x)
/// Uses the three-term recurrence relation for numerical evaluation.
pub fn jacobi_p_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 4 {
    return Err(InterpreterError::EvaluationError(
      "JacobiP expects exactly 4 arguments".into(),
    ));
  }

  let n = match &args[0] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "JacobiP".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Try numerical evaluation
  let a_f = try_eval_to_f64(&args[1]);
  let b_f = try_eval_to_f64(&args[2]);
  let x_f = try_eval_to_f64(&args[3]);

  if let (Some(a), Some(b), Some(x)) = (a_f, b_f, x_f) {
    let result = jacobi_p_f64(n, a, b, x);
    return Ok(Expr::Real(result));
  }

  // Try rational evaluation for integer/rational a, b, x
  // For now, return unevaluated for symbolic args
  Ok(Expr::FunctionCall {
    name: "JacobiP".to_string(),
    args: args.to_vec(),
  })
}

/// Evaluate the Jacobi polynomial P_n^{(a,b)}(x) numerically using
/// the three-term recurrence relation.
pub fn jacobi_p_f64(n: usize, a: f64, b: f64, x: f64) -> f64 {
  if n == 0 {
    return 1.0;
  }
  // P_1^{(a,b)}(x) = (a - b)/2 + (a + b + 2)*x/2
  let p1 = (a - b) / 2.0 + (a + b + 2.0) * x / 2.0;
  if n == 1 {
    return p1;
  }

  let mut prev = 1.0; // P_0
  let mut curr = p1; // P_1

  for k in 1..n {
    let k_f = k as f64;
    let n_f = k_f + 1.0; // n in recurrence (computing P_{k+1})
    let ab = a + b;
    let two_n = 2.0 * n_f;
    let denom = 2.0 * n_f * (n_f + ab) * (two_n + ab - 2.0);
    if denom.abs() < 1e-300 {
      break;
    }
    let a1 = (two_n + ab - 1.0)
      * ((two_n + ab) * (two_n + ab - 2.0) * x + a * a - b * b);
    let a2 = 2.0 * (n_f + a - 1.0) * (n_f + b - 1.0) * (two_n + ab);
    let next = (a1 * curr - a2 * prev) / denom;
    prev = curr;
    curr = next;
  }
  curr
}

/// LegendreP[n, x] - Legendre polynomial of degree n
pub fn legendre_p_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LegendreP expects exactly 2 arguments".into(),
    ));
  }

  let n = match &args[0] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "LegendreP".to_string(),
        args: args.to_vec(),
      });
    }
  };

  match &args[1] {
    Expr::Integer(x) => {
      // Evaluate at integer x using recurrence with rationals
      let (num, den) = legendre_eval_rational(n, (*x, 1));
      Ok(make_rational(num, den))
    }
    Expr::FunctionCall {
      name,
      args: rat_args,
    } if name == "Rational" && rat_args.len() == 2 => {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&rat_args[0], &rat_args[1])
      {
        let (num, den) = legendre_eval_rational(n, (*p, *q));
        Ok(make_rational(num, den))
      } else {
        Ok(Expr::FunctionCall {
          name: "LegendreP".to_string(),
          args: args.to_vec(),
        })
      }
    }
    Expr::Real(f) => {
      // Numeric evaluation using recurrence
      Ok(Expr::Real(legendre_eval_f64(n, *f)))
    }
    _ => {
      // Symbolic: build the polynomial expression
      if let Some(expr) = legendre_polynomial_symbolic(n, &args[1]) {
        Ok(expr)
      } else {
        Ok(Expr::FunctionCall {
          name: "LegendreP".to_string(),
          args: args.to_vec(),
        })
      }
    }
  }
}

/// Evaluate P_n(p/q) as a rational number using the recurrence
pub fn legendre_eval_rational(n: usize, x: (i128, i128)) -> (i128, i128) {
  let (xn, xd) = x;
  if n == 0 {
    return (1, 1);
  }
  if n == 1 {
    return (xn, xd);
  }

  let mut prev_n: i128 = 1; // P_0 = 1/1
  let mut prev_d: i128 = 1;
  let mut curr_n: i128 = xn; // P_1 = x
  let mut curr_d: i128 = xd;

  for m in 1..n {
    // P_{m+1} = ((2m+1)*x*P_m - m*P_{m-1}) / (m+1)
    let m_i = m as i128;

    // (2m+1)*x*P_m = (2m+1) * xn/xd * curr_n/curr_d
    let term1_n = (2 * m_i + 1) * xn * curr_n;
    let term1_d = xd * curr_d;

    // m*P_{m-1} = m * prev_n/prev_d
    let term2_n = m_i * prev_n;
    let term2_d = prev_d;

    // (term1 - term2) / (m+1)
    let diff_n = term1_n * term2_d - term2_n * term1_d;
    let diff_d = term1_d * term2_d * (m_i + 1);

    let g = gcd(diff_n.abs(), diff_d.abs());
    let next_n = diff_n / g;
    let next_d = diff_d / g;

    prev_n = curr_n;
    prev_d = curr_d;
    curr_n = next_n;
    curr_d = next_d;
  }

  (curr_n, curr_d)
}

/// Evaluate P_n(x) numerically using the recurrence
pub fn legendre_eval_f64(n: usize, x: f64) -> f64 {
  if n == 0 {
    return 1.0;
  }
  if n == 1 {
    return x;
  }

  let mut prev = 1.0;
  let mut curr = x;
  for m in 1..n {
    let next =
      ((2 * m + 1) as f64 * x * curr - m as f64 * prev) / (m + 1) as f64;
    prev = curr;
    curr = next;
  }
  curr
}

/// Compute the associated Legendre polynomial P_l^m(x) numerically.
/// Uses the recurrence relation starting from P_m^m and P_{m+1}^m.
pub fn associated_legendre_f64(l: i64, m: i64, x: f64) -> f64 {
  let m_abs = m.unsigned_abs() as usize;
  let l = l as usize;

  if m_abs > l {
    return 0.0;
  }

  // Compute P_m^m(x) = (-1)^m * (2m-1)!! * (1-x^2)^(m/2)
  let sin_theta = (1.0 - x * x).max(0.0).sqrt();
  let mut pmm = 1.0;
  for i in 1..=m_abs {
    pmm *= -(2.0 * i as f64 - 1.0) * sin_theta;
  }

  if l == m_abs {
    if m < 0 {
      // P_l^{-m}(x) = (-1)^m * (l-m)!/(l+m)! * P_l^m(x)
      let sign = if m_abs.is_multiple_of(2) { 1.0 } else { -1.0 };
      let mut ratio = 1.0;
      for i in (l - m_abs + 1)..=(l + m_abs) {
        ratio *= i as f64;
      }
      return sign * pmm / ratio;
    }
    return pmm;
  }

  // Compute P_{m+1}^m(x) = x * (2m+1) * P_m^m(x)
  let mut pmm1 = x * (2.0 * m_abs as f64 + 1.0) * pmm;

  if l == m_abs + 1 {
    if m < 0 {
      let sign = if m_abs.is_multiple_of(2) { 1.0 } else { -1.0 };
      let mut ratio = 1.0;
      for i in (l - m_abs + 1)..=(l + m_abs) {
        ratio *= i as f64;
      }
      return sign * pmm1 / ratio;
    }
    return pmm1;
  }

  // Recurrence: (l-m)*P_l^m = x*(2l-1)*P_{l-1}^m - (l+m-1)*P_{l-2}^m
  let mut result = 0.0;
  for ll in (m_abs + 2)..=l {
    result = (x * (2.0 * ll as f64 - 1.0) * pmm1
      - (ll + m_abs - 1) as f64 * pmm)
      / (ll - m_abs) as f64;
    pmm = pmm1;
    pmm1 = result;
  }

  if m < 0 {
    let sign = if m_abs.is_multiple_of(2) { 1.0 } else { -1.0 };
    let mut ratio = 1.0;
    for i in (l - m_abs + 1)..=(l + m_abs) {
      ratio *= i as f64;
    }
    result * sign / ratio
  } else {
    result
  }
}

/// SphericalHarmonicY[l, m, theta, phi] - Spherical harmonic function
pub fn spherical_harmonic_y_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 4 {
    return Err(InterpreterError::EvaluationError(
      "SphericalHarmonicY expects exactly 4 arguments".into(),
    ));
  }

  // Try to get integer values for l and m
  let l_val = match &args[0] {
    Expr::Integer(n) => Some(*n as i64),
    _ => None,
  };
  let m_val = match &args[1] {
    Expr::Integer(n) => Some(*n as i64),
    _ => None,
  };

  let (l, m) = match (l_val, m_val) {
    (Some(l), Some(m)) => (l, m),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "SphericalHarmonicY".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // |m| > l → 0
  if m.unsigned_abs() > l.unsigned_abs() {
    return Ok(Expr::Integer(0));
  }

  // l < 0 → undefined, return unevaluated
  if l < 0 {
    return Ok(Expr::FunctionCall {
      name: "SphericalHarmonicY".to_string(),
      args: args.to_vec(),
    });
  }

  // Try numerical evaluation
  let theta_f = try_eval_to_f64(&args[2]);
  let phi_f = try_eval_to_f64(&args[3]);

  if let (Some(theta), Some(phi)) = (theta_f, phi_f) {
    let cos_theta = theta.cos();
    let m_abs = m.unsigned_abs() as usize;
    let l_u = l as usize;

    // Normalization factor: sqrt((2l+1)/(4π) * (l-|m|)!/(l+|m|)!)
    let mut fact_ratio = 1.0_f64;
    for i in (l_u - m_abs + 1)..=(l_u + m_abs) {
      fact_ratio *= i as f64;
    }
    let norm =
      ((2.0 * l as f64 + 1.0) / (4.0 * std::f64::consts::PI) / fact_ratio)
        .sqrt();

    // Associated Legendre polynomial P_l^|m|(cos θ)
    let plm = associated_legendre_f64(l, m.abs(), cos_theta);

    // Condon-Shortley phase: (-1)^m for m > 0
    let cs_phase = if m > 0 && m % 2 != 0 { -1.0 } else { 1.0 };

    // Y_l^m = cs_phase * norm * P_l^|m|(cos θ) * e^(imφ)
    let re = cs_phase * norm * plm * (m as f64 * phi).cos();
    let im = cs_phase * norm * plm * (m as f64 * phi).sin();

    if im.abs() < 1e-15 {
      return Ok(Expr::Real(re));
    }
    return Ok(build_complex_float_expr(re, im));
  }

  // Return unevaluated for symbolic arguments
  Ok(Expr::FunctionCall {
    name: "SphericalHarmonicY".to_string(),
    args: args.to_vec(),
  })
}

/// Build the symbolic Legendre polynomial expression for P_n(x)
pub fn legendre_polynomial_symbolic(n: usize, x: &Expr) -> Option<Expr> {
  if n == 0 {
    return Some(Expr::Integer(1));
  }
  if n == 1 {
    return Some(x.clone());
  }

  // Compute polynomial coefficients as rationals
  let coeffs = legendre_coefficients(n)?;

  // Find LCM of all denominators
  let mut lcm: i128 = 1;
  for &(_, d) in &coeffs {
    if d != 0 {
      lcm = lcm_i128(lcm, d);
    }
  }

  // Build integer coefficients: int_coeff[k] = coeff[k] * lcm
  let mut int_coeffs: Vec<i128> = Vec::new();
  for &(cn, cd) in &coeffs {
    int_coeffs.push(cn * (lcm / cd));
  }

  // Build the polynomial sum: int_coeff_0 + int_coeff_1*x + int_coeff_2*x^2 + ...
  let mut terms: Vec<Expr> = Vec::new();
  for (k, &c) in int_coeffs.iter().enumerate() {
    if c == 0 {
      continue;
    }
    let term = if k == 0 {
      Expr::Integer(c)
    } else {
      let x_power = if k == 1 {
        x.clone()
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(x.clone()),
          right: Box::new(Expr::Integer(k as i128)),
        }
      };
      if c == 1 {
        x_power
      } else if c == -1 {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(-1)),
          right: Box::new(x_power),
        }
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(c)),
          right: Box::new(x_power),
        }
      }
    };
    terms.push(term);
  }

  let numerator = if terms.len() == 1 {
    terms.into_iter().next().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms,
    }
  };

  if lcm == 1 {
    Some(numerator)
  } else {
    Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(numerator),
      right: Box::new(Expr::Integer(lcm)),
    })
  }
}

/// Compute Legendre polynomial coefficients using the recurrence relation.
/// Returns coefficients [a_0, a_1, ..., a_n] as (numerator, denominator) pairs.
pub fn legendre_coefficients(n: usize) -> Option<Vec<(i128, i128)>> {
  if n == 0 {
    return Some(vec![(1, 1)]);
  }
  if n == 1 {
    return Some(vec![(0, 1), (1, 1)]);
  }

  let mut prev = vec![(1_i128, 1_i128)]; // P_0
  let mut curr = vec![(0_i128, 1_i128), (1_i128, 1_i128)]; // P_1

  for m in 1..n {
    let m_i = m as i128;
    let mut next = vec![(0_i128, 1_i128); m + 2];

    // (2m+1)*x*P_m(x): shift curr right by 1 and multiply by (2m+1)
    for (k, &(cn, cd)) in curr.iter().enumerate() {
      if cn == 0 {
        continue;
      }
      let term_n = (2 * m_i + 1).checked_mul(cn)?;
      let (nn, nd) = next[k + 1];
      let new_n = nn.checked_mul(cd)?.checked_add(term_n.checked_mul(nd)?)?;
      let new_d = nd.checked_mul(cd)?;
      let g = gcd(new_n.abs(), new_d.abs());
      next[k + 1] = (new_n / g, new_d / g);
    }

    // -m*P_{m-1}(x)
    for (k, &(cn, cd)) in prev.iter().enumerate() {
      if cn == 0 {
        continue;
      }
      let term_n = (-m_i).checked_mul(cn)?;
      let (nn, nd) = next[k];
      let new_n = nn.checked_mul(cd)?.checked_add(term_n.checked_mul(nd)?)?;
      let new_d = nd.checked_mul(cd)?;
      let g = gcd(new_n.abs(), new_d.abs());
      next[k] = (new_n / g, new_d / g);
    }

    // Divide by (m+1)
    for coeff in next.iter_mut() {
      if coeff.0 == 0 {
        continue;
      }
      let new_d = coeff.1.checked_mul(m_i + 1)?;
      let g = gcd(coeff.0.abs(), new_d.abs());
      *coeff = (coeff.0 / g, new_d / g);
    }

    prev = curr;
    curr = next;
  }

  Some(curr)
}

pub fn lcm_i128(a: i128, b: i128) -> i128 {
  let g = gcd(a.abs(), b.abs());
  if g == 0 {
    return 0;
  }
  (a.abs() / g) * b.abs()
}

/// LegendreQ[n, x] - Legendre function of the second kind
pub fn legendre_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LegendreQ expects exactly 2 arguments".into(),
    ));
  }

  let n = match &args[0] {
    Expr::Integer(n) if *n >= 0 => Some(*n as usize),
    Expr::Real(f) if *f >= 0.0 && *f == f.floor() => Some(*f as usize),
    _ => None,
  };

  let n = match n {
    Some(n) => n,
    None => {
      return Ok(Expr::FunctionCall {
        name: "LegendreQ".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Numeric evaluation
  if let Some(x_f) = expr_to_f64(&args[1])
    && (matches!(&args[1], Expr::Real(_)) || matches!(&args[0], Expr::Real(_)))
  {
    return Ok(Expr::Real(legendre_q_eval_f64(n, x_f)));
  }

  // Return unevaluated for symbolic
  Ok(Expr::FunctionCall {
    name: "LegendreQ".to_string(),
    args: args.to_vec(),
  })
}

/// Evaluate Q_n(x) numerically using recurrence
/// Q_0(x) = (1/2)*ln((1+x)/(1-x)), Q_1(x) = x*Q_0(x) - 1
/// (n+1)*Q_{n+1}(x) = (2n+1)*x*Q_n(x) - n*Q_{n-1}(x)
pub fn legendre_q_eval_f64(n: usize, x: f64) -> f64 {
  let q0 = 0.5 * ((1.0 + x) / (1.0 - x)).ln();
  if n == 0 {
    return q0;
  }
  let q1 = x * q0 - 1.0;
  if n == 1 {
    return q1;
  }

  let mut prev = q0;
  let mut curr = q1;
  for m in 1..n {
    let mf = m as f64;
    let next = ((2.0 * mf + 1.0) * x * curr - mf * prev) / (mf + 1.0);
    prev = curr;
    curr = next;
  }
  curr
}

/// ChebyshevT[n, x] - Chebyshev polynomial of the first kind
pub fn chebyshev_t_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ChebyshevT expects exactly 2 arguments".into(),
    ));
  }

  let n = match &args[0] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ChebyshevT".to_string(),
        args: args.to_vec(),
      });
    }
  };

  match &args[1] {
    Expr::Integer(x) => {
      let (num, den) = chebyshev_t_eval_rational(n, (*x, 1));
      Ok(make_rational(num, den))
    }
    Expr::FunctionCall {
      name,
      args: rat_args,
    } if name == "Rational" && rat_args.len() == 2 => {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&rat_args[0], &rat_args[1])
      {
        let (num, den) = chebyshev_t_eval_rational(n, (*p, *q));
        Ok(make_rational(num, den))
      } else {
        Ok(Expr::FunctionCall {
          name: "ChebyshevT".to_string(),
          args: args.to_vec(),
        })
      }
    }
    Expr::Real(f) => Ok(Expr::Real(chebyshev_t_eval_f64(n, *f))),
    _ => {
      // Symbolic: build the polynomial expression
      if let Some(expr) = chebyshev_t_polynomial_symbolic(n, &args[1]) {
        Ok(expr)
      } else {
        Ok(Expr::FunctionCall {
          name: "ChebyshevT".to_string(),
          args: args.to_vec(),
        })
      }
    }
  }
}

/// Evaluate T_n(p/q) as a rational number using the recurrence
/// T_0(x) = 1, T_1(x) = x, T_{n+1}(x) = 2x*T_n(x) - T_{n-1}(x)
pub fn chebyshev_t_eval_rational(n: usize, x: (i128, i128)) -> (i128, i128) {
  let (xn, xd) = x;
  if n == 0 {
    return (1, 1);
  }
  if n == 1 {
    return (xn, xd);
  }

  let mut tm1 = (1i128, 1i128); // T_0
  let mut t = (xn, xd); // T_1

  for _ in 2..=n {
    // T_{k} = 2x * T_{k-1} - T_{k-2}
    // 2x * T_{k-1}: (2 * xn * t.0, xd * t.1)
    let a_n = 2i128
      .checked_mul(xn)
      .and_then(|v| v.checked_mul(t.0))
      .unwrap_or(0);
    let a_d = xd.checked_mul(t.1).unwrap_or(1);

    // a - tm1: (a_n * tm1.1 - tm1.0 * a_d, a_d * tm1.1)
    let new_n = a_n
      .checked_mul(tm1.1)
      .and_then(|v| v.checked_sub(tm1.0.checked_mul(a_d)?))
      .unwrap_or(0);
    let new_d = a_d.checked_mul(tm1.1).unwrap_or(1);

    let g = gcd(new_n.abs(), new_d.abs());
    tm1 = t;
    t = if g > 0 {
      (new_n / g, new_d / g)
    } else {
      (new_n, new_d)
    };
  }
  t
}

/// Evaluate T_n(x) numerically
pub fn chebyshev_t_eval_f64(n: usize, x: f64) -> f64 {
  if n == 0 {
    return 1.0;
  }
  if n == 1 {
    return x;
  }

  let mut tm1 = 1.0;
  let mut t = x;
  for _ in 2..=n {
    let tnew = 2.0 * x * t - tm1;
    tm1 = t;
    t = tnew;
  }
  t
}

/// Build symbolic Chebyshev polynomial T_n(x)
/// T_n has coefficients that can be computed via recurrence
pub fn chebyshev_t_polynomial_symbolic(n: usize, x: &Expr) -> Option<Expr> {
  // Compute coefficients: T_n(x) = Σ c_k x^k
  let coeffs = chebyshev_t_coefficients(n)?;

  // Build expression as sum of terms
  let mut terms: Vec<Expr> = Vec::new();
  for (k, (cn, cd)) in coeffs.iter().enumerate() {
    if *cn == 0 {
      continue;
    }
    let coeff = (*cn, *cd);
    let x_power = if k == 0 {
      None
    } else if k == 1 {
      Some(x.clone())
    } else {
      Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(x.clone()),
        right: Box::new(Expr::Integer(k as i128)),
      })
    };

    let term = match (coeff, x_power) {
      ((c, 1), None) => Expr::Integer(c),
      ((1, 1), Some(xp)) => xp,
      ((-1, 1), Some(xp)) => Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), xp],
      },
      ((c, 1), Some(xp)) => Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(c)),
        right: Box::new(xp),
      },
      _ => return None, // Should not happen for Chebyshev (all integer coefficients)
    };
    terms.push(term);
  }

  if terms.is_empty() {
    return Some(Expr::Integer(0));
  }
  if terms.len() == 1 {
    return Some(terms.pop().unwrap());
  }

  // Build sum from left to right using Plus
  let mut result = terms[0].clone();
  for t in terms.iter().skip(1) {
    result = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(result),
      right: Box::new(t.clone()),
    };
  }
  Some(result)
}

/// Compute Chebyshev T coefficients as (numerator, denominator) pairs
/// T_n(x) = Σ c_k x^k where all c_k are integers (denom = 1)
pub fn chebyshev_t_coefficients(n: usize) -> Option<Vec<(i128, i128)>> {
  if n == 0 {
    return Some(vec![(1, 1)]);
  }
  if n == 1 {
    return Some(vec![(0, 1), (1, 1)]);
  }

  let mut prev: Vec<i128> = vec![1]; // T_0 coefficients
  let mut curr: Vec<i128> = vec![0, 1]; // T_1 coefficients

  for _ in 2..=n {
    // T_{k} = 2x * T_{k-1} - T_{k-2}
    // 2x * curr: shift coefficients right and multiply by 2
    let mut next = vec![0i128; curr.len() + 1];
    for (j, c) in curr.iter().enumerate() {
      next[j + 1] = next[j + 1].checked_add(2i128.checked_mul(*c)?)?;
    }
    // Subtract prev
    for (j, c) in prev.iter().enumerate() {
      next[j] = next[j].checked_sub(*c)?;
    }

    prev = curr;
    curr = next;
  }

  Some(curr.into_iter().map(|c| (c, 1i128)).collect())
}

/// ChebyshevU[n, x] - Chebyshev polynomial of the second kind
pub fn chebyshev_u_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ChebyshevU expects exactly 2 arguments".into(),
    ));
  }

  let n = match &args[0] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ChebyshevU".to_string(),
        args: args.to_vec(),
      });
    }
  };

  match &args[1] {
    Expr::Integer(x) => {
      let (num, den) = chebyshev_u_eval_rational(n, (*x, 1));
      Ok(make_rational(num, den))
    }
    Expr::FunctionCall {
      name,
      args: rat_args,
    } if name == "Rational" && rat_args.len() == 2 => {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&rat_args[0], &rat_args[1])
      {
        let (num, den) = chebyshev_u_eval_rational(n, (*p, *q));
        Ok(make_rational(num, den))
      } else {
        Ok(Expr::FunctionCall {
          name: "ChebyshevU".to_string(),
          args: args.to_vec(),
        })
      }
    }
    Expr::Real(f) => Ok(Expr::Real(chebyshev_u_eval_f64(n, *f))),
    _ => {
      if let Some(expr) = chebyshev_u_polynomial_symbolic(n, &args[1]) {
        Ok(expr)
      } else {
        Ok(Expr::FunctionCall {
          name: "ChebyshevU".to_string(),
          args: args.to_vec(),
        })
      }
    }
  }
}

/// Evaluate U_n(p/q) as a rational number
/// U_0(x) = 1, U_1(x) = 2x, U_{n+1}(x) = 2x*U_n(x) - U_{n-1}(x)
pub fn chebyshev_u_eval_rational(n: usize, x: (i128, i128)) -> (i128, i128) {
  let (xn, xd) = x;
  if n == 0 {
    return (1, 1);
  }
  if n == 1 {
    return (2 * xn, xd);
  }

  let mut um1 = (1i128, 1i128); // U_0
  let mut u = (2 * xn, xd); // U_1

  for _ in 2..=n {
    // U_{k} = 2x * U_{k-1} - U_{k-2}
    let a_n = 2i128
      .checked_mul(xn)
      .and_then(|v| v.checked_mul(u.0))
      .unwrap_or(0);
    let a_d = xd.checked_mul(u.1).unwrap_or(1);

    let new_n = a_n
      .checked_mul(um1.1)
      .and_then(|v| v.checked_sub(um1.0.checked_mul(a_d)?))
      .unwrap_or(0);
    let new_d = a_d.checked_mul(um1.1).unwrap_or(1);

    let g = gcd(new_n.abs(), new_d.abs());
    um1 = u;
    u = if g > 0 {
      (new_n / g, new_d / g)
    } else {
      (new_n, new_d)
    };
  }
  u
}

/// Evaluate U_n(x) numerically
pub fn chebyshev_u_eval_f64(n: usize, x: f64) -> f64 {
  if n == 0 {
    return 1.0;
  }
  if n == 1 {
    return 2.0 * x;
  }

  let mut um1 = 1.0;
  let mut u = 2.0 * x;
  for _ in 2..=n {
    let unew = 2.0 * x * u - um1;
    um1 = u;
    u = unew;
  }
  u
}

/// Build symbolic Chebyshev U polynomial
pub fn chebyshev_u_polynomial_symbolic(n: usize, x: &Expr) -> Option<Expr> {
  let coeffs = chebyshev_u_coefficients(n)?;

  let mut terms: Vec<Expr> = Vec::new();
  for (k, (cn, cd)) in coeffs.iter().enumerate() {
    if *cn == 0 {
      continue;
    }
    let coeff = (*cn, *cd);
    let x_power = if k == 0 {
      None
    } else if k == 1 {
      Some(x.clone())
    } else {
      Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(x.clone()),
        right: Box::new(Expr::Integer(k as i128)),
      })
    };

    let term = match (coeff, x_power) {
      ((c, 1), None) => Expr::Integer(c),
      ((1, 1), Some(xp)) => xp,
      ((-1, 1), Some(xp)) => Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), xp],
      },
      ((c, 1), Some(xp)) => Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(c)),
        right: Box::new(xp),
      },
      _ => return None,
    };
    terms.push(term);
  }

  if terms.is_empty() {
    return Some(Expr::Integer(0));
  }
  if terms.len() == 1 {
    return Some(terms.pop().unwrap());
  }

  let mut result = terms[0].clone();
  for t in terms.iter().skip(1) {
    result = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(result),
      right: Box::new(t.clone()),
    };
  }
  Some(result)
}

/// Compute Chebyshev U coefficients
/// U_0 = [1], U_1 = [0, 2], U_{n+1} = 2x*U_n - U_{n-1}
pub fn chebyshev_u_coefficients(n: usize) -> Option<Vec<(i128, i128)>> {
  if n == 0 {
    return Some(vec![(1, 1)]);
  }
  if n == 1 {
    return Some(vec![(0, 1), (2, 1)]);
  }

  let mut prev: Vec<i128> = vec![1]; // U_0
  let mut curr: Vec<i128> = vec![0, 2]; // U_1

  for _ in 2..=n {
    let mut next = vec![0i128; curr.len() + 1];
    for (j, c) in curr.iter().enumerate() {
      next[j + 1] = next[j + 1].checked_add(2i128.checked_mul(*c)?)?;
    }
    for (j, c) in prev.iter().enumerate() {
      next[j] = next[j].checked_sub(*c)?;
    }

    prev = curr;
    curr = next;
  }

  Some(curr.into_iter().map(|c| (c, 1i128)).collect())
}

/// GegenbauerC[n, lambda, x] - Gegenbauer (ultraspherical) polynomial
pub fn gegenbauer_c_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "GegenbauerC expects exactly 3 arguments".into(),
    ));
  }

  let n = match &args[0] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "GegenbauerC".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Extract lambda as rational (p/q)
  let lambda = match &args[1] {
    Expr::Integer(v) => Some((*v, 1i128)),
    Expr::FunctionCall {
      name,
      args: rat_args,
    } if name == "Rational" && rat_args.len() == 2 => {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&rat_args[0], &rat_args[1])
      {
        Some((*p, *q))
      } else {
        None
      }
    }
    _ => None,
  };

  // If lambda is not rational, check if x is real for numeric eval
  if lambda.is_none() {
    if let Some(lam_f) = expr_to_f64(&args[1])
      && let Some(x_f) = expr_to_f64(&args[2])
      && (matches!(&args[1], Expr::Real(_))
        || matches!(&args[2], Expr::Real(_)))
    {
      return Ok(Expr::Real(gegenbauer_eval_f64(n, lam_f, x_f)));
    }
    return Ok(Expr::FunctionCall {
      name: "GegenbauerC".to_string(),
      args: args.to_vec(),
    });
  }

  let lam = lambda.unwrap();

  match &args[2] {
    Expr::Integer(x) => {
      let (num, den) = gegenbauer_eval_rational(n, lam, (*x, 1));
      Ok(make_rational(num, den))
    }
    Expr::FunctionCall {
      name,
      args: rat_args,
    } if name == "Rational" && rat_args.len() == 2 => {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&rat_args[0], &rat_args[1])
      {
        let (num, den) = gegenbauer_eval_rational(n, lam, (*p, *q));
        Ok(make_rational(num, den))
      } else {
        Ok(Expr::FunctionCall {
          name: "GegenbauerC".to_string(),
          args: args.to_vec(),
        })
      }
    }
    Expr::Real(f) => {
      let lam_f = lam.0 as f64 / lam.1 as f64;
      Ok(Expr::Real(gegenbauer_eval_f64(n, lam_f, *f)))
    }
    _ => {
      // Symbolic: build polynomial
      if let Some(expr) = gegenbauer_polynomial_symbolic(n, lam, &args[2]) {
        Ok(expr)
      } else {
        Ok(Expr::FunctionCall {
          name: "GegenbauerC".to_string(),
          args: args.to_vec(),
        })
      }
    }
  }
}

/// Evaluate Gegenbauer C_n^lambda(p/q) as rational
/// C_0^λ = 1, C_1^λ = 2λx, C_{k+1}^λ = (2(k+λ)x C_k^λ - (k+2λ-1) C_{k-1}^λ) / (k+1)
pub fn gegenbauer_eval_rational(
  n: usize,
  lam: (i128, i128),
  x: (i128, i128),
) -> (i128, i128) {
  if n == 0 {
    return (1, 1);
  }
  // C_1 = 2λx = (2*lam_n*x_n, lam_d*x_d)
  let c1_n = 2i128
    .checked_mul(lam.0)
    .and_then(|v| v.checked_mul(x.0))
    .unwrap_or(0);
  let c1_d = lam.1.checked_mul(x.1).unwrap_or(1);
  let g = gcd(c1_n.abs(), c1_d.abs());
  let c1 = if g > 0 {
    (c1_n / g, c1_d / g)
  } else {
    (c1_n, c1_d)
  };
  if n == 1 {
    return c1;
  }

  let mut cm1 = (1i128, 1i128);
  let mut c = c1;

  for k in 1..n {
    let kk = k as i128;
    // coeff_a = 2(k + λ) = 2*(k*lam_d + lam_n)/lam_d
    let k_plus_lam_n = kk
      .checked_mul(lam.1)
      .and_then(|v| v.checked_add(lam.0))
      .unwrap_or(0);
    let k_plus_lam_d = lam.1;
    // 2*(k+λ)*x * C_k: numerator = 2 * k_plus_lam_n * x_n * c_n
    let a_n = 2i128
      .checked_mul(k_plus_lam_n)
      .and_then(|v| v.checked_mul(x.0))
      .and_then(|v| v.checked_mul(c.0))
      .unwrap_or(0);
    let a_d = k_plus_lam_d
      .checked_mul(x.1)
      .and_then(|v| v.checked_mul(c.1))
      .unwrap_or(1);

    // coeff_b = (k + 2λ - 1) = (k*lam_d + 2*lam_n - lam_d)/lam_d
    let b_n = kk
      .checked_mul(lam.1)
      .and_then(|v| v.checked_add(2i128.checked_mul(lam.0)?))
      .and_then(|v| v.checked_sub(lam.1))
      .unwrap_or(0);
    let b_d = lam.1;

    // b * C_{k-1}: b_n * cm1_n / (b_d * cm1_d)
    let sub_n = b_n.checked_mul(cm1.0).unwrap_or(0);
    let sub_d = b_d.checked_mul(cm1.1).unwrap_or(1);

    // (a - sub) / (k+1)
    // a/a_d - sub/sub_d = (a*sub_d - sub*a_d) / (a_d * sub_d)
    let diff_n = a_n
      .checked_mul(sub_d)
      .and_then(|v| v.checked_sub(sub_n.checked_mul(a_d)?))
      .unwrap_or(0);
    let diff_d = a_d.checked_mul(sub_d).unwrap_or(1);

    // Divide by (k+1)
    let new_n = diff_n;
    let new_d = diff_d.checked_mul(kk + 1).unwrap_or(1);

    let g = gcd(new_n.abs(), new_d.abs());
    cm1 = c;
    c = if g > 0 {
      (new_n / g, new_d / g)
    } else {
      (new_n, new_d)
    };
  }
  c
}

/// Evaluate Gegenbauer C_n^lambda(x) numerically
pub fn gegenbauer_eval_f64(n: usize, lam: f64, x: f64) -> f64 {
  if n == 0 {
    return 1.0;
  }
  if n == 1 {
    return 2.0 * lam * x;
  }

  let mut cm1 = 1.0;
  let mut c = 2.0 * lam * x;
  for k in 1..n {
    let kf = k as f64;
    let c_new =
      (2.0 * (kf + lam) * x * c - (kf + 2.0 * lam - 1.0) * cm1) / (kf + 1.0);
    cm1 = c;
    c = c_new;
  }
  c
}

/// Build symbolic Gegenbauer polynomial
pub fn gegenbauer_polynomial_symbolic(
  n: usize,
  lam: (i128, i128),
  x: &Expr,
) -> Option<Expr> {
  let coeffs = gegenbauer_coefficients(n, lam)?;

  let mut terms: Vec<Expr> = Vec::new();
  for (k, (cn, cd)) in coeffs.iter().enumerate() {
    if *cn == 0 {
      continue;
    }
    let x_power = if k == 0 {
      None
    } else if k == 1 {
      Some(x.clone())
    } else {
      Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(x.clone()),
        right: Box::new(Expr::Integer(k as i128)),
      })
    };

    let coeff_expr = if *cd == 1 {
      Expr::Integer(*cn)
    } else {
      make_rational(*cn, *cd)
    };

    let term = match x_power {
      None => coeff_expr,
      Some(xp) => {
        if *cn == 1 && *cd == 1 {
          xp
        } else if *cn == -1 && *cd == 1 {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), xp],
          }
        } else {
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(coeff_expr),
            right: Box::new(xp),
          }
        }
      }
    };
    terms.push(term);
  }

  if terms.is_empty() {
    return Some(Expr::Integer(0));
  }
  if terms.len() == 1 {
    return Some(terms.pop().unwrap());
  }

  let mut result = terms[0].clone();
  for t in terms.iter().skip(1) {
    result = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(result),
      right: Box::new(t.clone()),
    };
  }
  Some(result)
}

/// Compute Gegenbauer polynomial coefficients as (numerator, denominator) pairs
pub fn gegenbauer_coefficients(
  n: usize,
  lam: (i128, i128),
) -> Option<Vec<(i128, i128)>> {
  if n == 0 {
    return Some(vec![(1, 1)]);
  }
  if n == 1 {
    // 2λx: coefficient of x^1 is 2λ = 2*lam_n/lam_d
    let cn = 2i128.checked_mul(lam.0)?;
    let g = gcd(cn.abs(), lam.1.abs());
    return Some(vec![(0, 1), (cn / g, lam.1 / g)]);
  }

  // Store coefficients as (numerator, denominator) vectors
  let mut prev: Vec<(i128, i128)> = vec![(1, 1)]; // C_0
  let cn = 2i128.checked_mul(lam.0)?;
  let g = gcd(cn.abs(), lam.1.abs());
  let mut curr: Vec<(i128, i128)> = vec![(0, 1), (cn / g, lam.1 / g)]; // C_1

  for k in 1..n {
    let kk = k as i128;
    // C_{k+1} = (2(k+λ)x * C_k - (k+2λ-1) * C_{k-1}) / (k+1)
    // coeff_a = 2(k+λ) = 2*(k*lam_d + lam_n) / lam_d
    let a_n = 2i128.checked_mul(kk.checked_mul(lam.1)?.checked_add(lam.0)?)?;
    let a_d = lam.1;

    // coeff_b = (k + 2λ - 1) = (k*lam_d + 2*lam_n - lam_d) / lam_d
    let b_n = kk
      .checked_mul(lam.1)?
      .checked_add(2i128.checked_mul(lam.0)?)?
      .checked_sub(lam.1)?;
    let b_d = lam.1;

    // 2(k+λ)x * curr: shift right and multiply by a_n/a_d
    let mut next: Vec<(i128, i128)> = vec![(0, 1); curr.len() + 1];
    for (j, (cn, cd)) in curr.iter().enumerate() {
      // a_n/a_d * cn/cd = a_n*cn / (a_d*cd)
      let nn = a_n.checked_mul(*cn)?;
      let nd = a_d.checked_mul(*cd)?;
      let g = gcd(nn.abs(), nd.abs());
      next[j + 1] = if g > 0 { (nn / g, nd / g) } else { (nn, nd) };
    }

    // Subtract b * prev
    for (j, (pn, pd)) in prev.iter().enumerate() {
      // Subtract b_n/b_d * pn/pd
      let sub_n = b_n.checked_mul(*pn)?;
      let sub_d = b_d.checked_mul(*pd)?;
      // next[j] = next[j] - sub_n/sub_d
      let (ref nn, ref nd) = next[j];
      // nn/nd - sub_n/sub_d = (nn*sub_d - sub_n*nd) / (nd*sub_d)
      let res_n = nn
        .checked_mul(sub_d)?
        .checked_sub(sub_n.checked_mul(*nd)?)?;
      let res_d = nd.checked_mul(sub_d)?;
      let g = gcd(res_n.abs(), res_d.abs());
      next[j] = if g > 0 {
        (res_n / g, res_d / g)
      } else {
        (res_n, res_d)
      };
    }

    // Divide by (k+1)
    let div = kk + 1;
    for (cn, cd) in next.iter_mut() {
      *cd = cd.checked_mul(div)?;
      let g = gcd(cn.abs(), cd.abs());
      if g > 0 {
        *cn /= g;
        *cd /= g;
      }
    }

    prev = curr;
    curr = next;
  }

  Some(curr)
}

/// LaguerreL[n, x] - Laguerre polynomial
pub fn laguerre_l_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LaguerreL expects exactly 2 arguments".into(),
    ));
  }

  let n = match &args[0] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "LaguerreL".to_string(),
        args: args.to_vec(),
      });
    }
  };

  match &args[1] {
    Expr::Integer(x) => {
      let (num, den) = laguerre_eval_rational(n, (*x, 1));
      Ok(make_rational(num, den))
    }
    Expr::FunctionCall {
      name,
      args: rat_args,
    } if name == "Rational" && rat_args.len() == 2 => {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&rat_args[0], &rat_args[1])
      {
        let (num, den) = laguerre_eval_rational(n, (*p, *q));
        Ok(make_rational(num, den))
      } else {
        Ok(Expr::FunctionCall {
          name: "LaguerreL".to_string(),
          args: args.to_vec(),
        })
      }
    }
    Expr::Real(f) => Ok(Expr::Real(laguerre_eval_f64(n, *f))),
    _ => {
      if let Some(expr) = laguerre_polynomial_symbolic(n, &args[1]) {
        Ok(expr)
      } else {
        Ok(Expr::FunctionCall {
          name: "LaguerreL".to_string(),
          args: args.to_vec(),
        })
      }
    }
  }
}

/// Evaluate L_n(p/q) using recurrence
pub fn laguerre_eval_rational(n: usize, x: (i128, i128)) -> (i128, i128) {
  let (xn, xd) = x;
  if n == 0 {
    return (1, 1);
  }
  if n == 1 {
    return (xd - xn, xd);
  }

  let mut lm1 = (1i128, 1i128);
  let mut l = (xd - xn, xd);

  for k in 1..n {
    let kf = k as i128;
    // L_{k+1} = ((2k+1-x)*L_k - k*L_{k-1}) / (k+1)
    let coeff_n = (2 * kf + 1)
      .checked_mul(xd)
      .and_then(|v| v.checked_sub(xn))
      .unwrap_or(0);
    let coeff_d = xd;

    let a_n = coeff_n.checked_mul(l.0).unwrap_or(0);
    let a_d = coeff_d.checked_mul(l.1).unwrap_or(1);

    let b_n = kf.checked_mul(lm1.0).unwrap_or(0);
    let b_d = lm1.1;

    let sub_n = a_n
      .checked_mul(b_d)
      .and_then(|v| v.checked_sub(b_n.checked_mul(a_d)?))
      .unwrap_or(0);
    let sub_d = a_d
      .checked_mul(b_d)
      .and_then(|v| v.checked_mul(kf + 1))
      .unwrap_or(1);

    let g = gcd(sub_n.abs(), sub_d.abs());
    lm1 = l;
    l = if g > 0 {
      (sub_n / g, sub_d / g)
    } else {
      (sub_n, sub_d)
    };
  }
  l
}

/// Evaluate L_n(x) numerically
pub fn laguerre_eval_f64(n: usize, x: f64) -> f64 {
  if n == 0 {
    return 1.0;
  }
  if n == 1 {
    return 1.0 - x;
  }

  let mut lm1 = 1.0;
  let mut l = 1.0 - x;
  for k in 1..n {
    let kf = k as f64;
    let lnew = ((2.0 * kf + 1.0 - x) * l - kf * lm1) / (kf + 1.0);
    lm1 = l;
    l = lnew;
  }
  l
}

/// Build symbolic Laguerre polynomial L_n(x)
/// Output as (c_0 + c_1*x + c_2*x^2 + ...) / n!
pub fn laguerre_polynomial_symbolic(n: usize, x: &Expr) -> Option<Expr> {
  let coeffs = laguerre_scaled_coefficients(n)?;
  let n_fact = factorial_i128(n)?;

  let mut terms: Vec<Expr> = Vec::new();
  for (k, c) in coeffs.iter().enumerate() {
    if *c == 0 {
      continue;
    }
    let x_power = if k == 0 {
      None
    } else if k == 1 {
      Some(x.clone())
    } else {
      Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(x.clone()),
        right: Box::new(Expr::Integer(k as i128)),
      })
    };

    let term = match x_power {
      None => Expr::Integer(*c),
      Some(xp) if *c == 1 => xp,
      Some(xp) if *c == -1 => Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), xp],
      },
      Some(xp) => Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(*c)),
        right: Box::new(xp),
      },
    };
    terms.push(term);
  }

  if terms.is_empty() {
    return Some(Expr::Integer(0));
  }

  let mut numerator = terms[0].clone();
  for t in terms.iter().skip(1) {
    numerator = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(numerator),
      right: Box::new(t.clone()),
    };
  }

  if n_fact == 1 {
    return Some(numerator);
  }

  Some(Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(numerator),
    right: Box::new(Expr::Integer(n_fact)),
  })
}

/// Compute Laguerre scaled coefficients: n! * L_n(x) = Σ c_k x^k
/// c_k = (-1)^k * C(n,k) * n! / k!
pub fn laguerre_scaled_coefficients(n: usize) -> Option<Vec<i128>> {
  let n_fact = factorial_i128(n)?;
  let mut coeffs = vec![0i128; n + 1];
  for k in 0..=n {
    let k_fact = factorial_i128(k)?;
    let nk_fact = factorial_i128(n - k)?;
    let binom = n_fact / (k_fact * nk_fact);
    let sign: i128 = if k % 2 == 0 { 1 } else { -1 };
    coeffs[k] = sign * binom * n_fact / k_fact;
  }
  Some(coeffs)
}

/// Compute n! as i128, returning None on overflow
pub fn factorial_i128(n: usize) -> Option<i128> {
  let mut result: i128 = 1;
  for i in 2..=n {
    result = result.checked_mul(i as i128)?;
  }
  Some(result)
}

/// HermiteH[n, x] - Hermite polynomial (physicist's convention)
/// H_0(x) = 1, H_1(x) = 2x, H_{n+1}(x) = 2x*H_n(x) - 2n*H_{n-1}(x)
pub fn hermite_h_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "HermiteH expects exactly 2 arguments".into(),
    ));
  }

  let n = match &args[0] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "HermiteH".to_string(),
        args: args.to_vec(),
      });
    }
  };

  match &args[1] {
    Expr::Integer(x) => {
      // Evaluate using recurrence with exact arithmetic
      let result = hermite_eval_i128(n, *x);
      Ok(Expr::Integer(result))
    }
    Expr::Real(f) => Ok(Expr::Real(hermite_eval_f64(n, *f))),
    _ => {
      if let Some(expr) = hermite_polynomial_symbolic(n, &args[1]) {
        Ok(expr)
      } else {
        Ok(Expr::FunctionCall {
          name: "HermiteH".to_string(),
          args: args.to_vec(),
        })
      }
    }
  }
}

/// Evaluate H_n(x) for integer x using exact i128 arithmetic
pub fn hermite_eval_i128(n: usize, x: i128) -> i128 {
  if n == 0 {
    return 1;
  }
  if n == 1 {
    return 2 * x;
  }
  let mut hm1: i128 = 1;
  let mut h: i128 = 2 * x;
  for k in 1..n {
    let hnew = 2 * x * h - 2 * (k as i128) * hm1;
    hm1 = h;
    h = hnew;
  }
  h
}

/// Evaluate H_n(x) numerically
pub fn hermite_eval_f64(n: usize, x: f64) -> f64 {
  if n == 0 {
    return 1.0;
  }
  if n == 1 {
    return 2.0 * x;
  }
  let mut hm1 = 1.0;
  let mut h = 2.0 * x;
  for k in 1..n {
    let hnew = 2.0 * x * h - 2.0 * (k as f64) * hm1;
    hm1 = h;
    h = hnew;
  }
  h
}

/// Build symbolic Hermite polynomial using coefficient recurrence
pub fn hermite_polynomial_symbolic(n: usize, x: &Expr) -> Option<Expr> {
  let coeffs = hermite_coefficients(n)?;

  let mut terms: Vec<Expr> = Vec::new();
  for (k, c) in coeffs.iter().enumerate() {
    if *c == 0 {
      continue;
    }
    let x_power = if k == 0 {
      None
    } else if k == 1 {
      Some(x.clone())
    } else {
      Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(x.clone()),
        right: Box::new(Expr::Integer(k as i128)),
      })
    };

    let term = match x_power {
      None => Expr::Integer(*c),
      Some(xp) if *c == 1 => xp,
      Some(xp) if *c == -1 => Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), xp],
      },
      Some(xp) => Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(*c)),
        right: Box::new(xp),
      },
    };
    terms.push(term);
  }

  if terms.is_empty() {
    return Some(Expr::Integer(0));
  }
  if terms.len() == 1 {
    return Some(terms.pop().unwrap());
  }

  let mut result = terms[0].clone();
  for t in terms.iter().skip(1) {
    result = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(result),
      right: Box::new(t.clone()),
    };
  }
  Some(result)
}

/// Compute Hermite polynomial coefficients H_n(x) = Σ c_k x^k
pub fn hermite_coefficients(n: usize) -> Option<Vec<i128>> {
  if n == 0 {
    return Some(vec![1]);
  }
  if n == 1 {
    return Some(vec![0, 2]);
  }

  let mut prev: Vec<i128> = vec![1];
  let mut curr: Vec<i128> = vec![0, 2];

  for k in 1..n {
    // H_{k+1} = 2x * H_k - 2k * H_{k-1}
    let mut next = vec![0i128; curr.len() + 1];
    for (j, c) in curr.iter().enumerate() {
      next[j + 1] = next[j + 1].checked_add(2i128.checked_mul(*c)?)?;
    }
    let kf = k as i128;
    for (j, c) in prev.iter().enumerate() {
      next[j] = next[j].checked_sub(2i128.checked_mul(kf)?.checked_mul(*c)?)?;
    }

    prev = curr;
    curr = next;
  }

  Some(curr)
}
