#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr};

/// PolyLog[s, z] - Polylogarithm function Li_s(z)
pub fn polylog_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "PolyLog expects exactly 2 arguments".into(),
    ));
  }

  let s_expr = &args[0];
  let z_expr = &args[1];

  // PolyLog[s, 0] = 0 for any s
  if matches!(z_expr, Expr::Integer(0)) {
    return Ok(Expr::Integer(0));
  }

  // PolyLog[s, 1] for integer s >= 2: return Zeta[s].
  // For symbolic s, Wolfram keeps PolyLog[s, 1] unevaluated — we match that.
  if matches!(z_expr, Expr::Integer(1))
    && let Expr::Integer(n) = s_expr
    && *n >= 2
  {
    return zeta_ast(&[Expr::Integer(*n)]);
  }

  match s_expr {
    Expr::Integer(s) => {
      return polylog_integer_s(*s, z_expr, args);
    }
    Expr::Real(sf) => {
      if let Some(zf) = extract_f64(z_expr) {
        return Ok(Expr::Real(polylog_numeric(*sf, zf)));
      }
    }
    _ => {}
  }

  Ok(Expr::FunctionCall {
    name: "PolyLog".to_string(),
    args: args.to_vec(),
  })
}

pub fn polylog_integer_s(
  s: i128,
  z_expr: &Expr,
  orig_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  // s = 1: PolyLog[1, z] = -Log[1-z]
  if s == 1 {
    return polylog_s1(z_expr);
  }

  // s = 0: PolyLog[0, z] = z/(1-z)
  if s == 0 {
    return polylog_s0(z_expr);
  }

  // s < 0: rational function via Eulerian numbers
  if s < 0 {
    return polylog_negative_s((-s) as usize, z_expr);
  }

  // s >= 2: special values at z = 0, 1, -1
  match z_expr {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Integer(1) => {
      // PolyLog[s, 1] = Zeta[s]
      return zeta_ast(&[Expr::Integer(s)]);
    }
    Expr::Integer(-1) => {
      // PolyLog[s, -1] = -(1 - 2^{1-s}) * Zeta[s]
      return polylog_at_neg1(s);
    }
    Expr::Real(f) => {
      return Ok(Expr::Real(polylog_numeric(s as f64, *f)));
    }
    _ => {}
  }

  Ok(Expr::FunctionCall {
    name: "PolyLog".to_string(),
    args: orig_args.to_vec(),
  })
}

/// PolyLog[1, z] = -Log[1-z]
pub fn polylog_s1(z_expr: &Expr) -> Result<Expr, InterpreterError> {
  match z_expr {
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    Expr::Integer(1) => Ok(Expr::Identifier("Infinity".to_string())),
    Expr::Integer(z) => {
      // 1-z is an integer, construct -Log[1-z]
      let one_minus_z = Expr::Integer(1 - z);
      Ok(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(Expr::FunctionCall {
          name: "Log".to_string(),
          args: vec![one_minus_z],
        }),
      })
    }
    Expr::FunctionCall {
      name,
      args: rat_args,
    } if name == "Rational" && rat_args.len() == 2 => {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&rat_args[0], &rat_args[1])
      {
        // 1 - p/q = (q-p)/q
        let one_minus_z = make_rational(q - p, *q);
        // Evaluate Log to get simplification (e.g., Log[1/2] → -Log[2])
        let log_val = log_ast(&[one_minus_z])?;
        // Negate: -Log[1 - p/q], simplifying double negation
        match &log_val {
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left,
            right,
          } if matches!(left.as_ref(), Expr::Integer(-1)) => {
            // -(-expr) = expr
            Ok(*right.clone())
          }
          _ => Ok(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(log_val),
          }),
        }
      } else {
        polylog_s1_symbolic(z_expr)
      }
    }
    Expr::Real(f) => {
      let result = -(1.0 - f).ln();
      Ok(Expr::Real(result))
    }
    _ => polylog_s1_symbolic(z_expr),
  }
}

pub fn polylog_s1_symbolic(z_expr: &Expr) -> Result<Expr, InterpreterError> {
  let one_minus_z = Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(Expr::Integer(1)),
    right: Box::new(z_expr.clone()),
  };
  Ok(Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(Expr::Integer(-1)),
    right: Box::new(Expr::FunctionCall {
      name: "Log".to_string(),
      args: vec![one_minus_z],
    }),
  })
}

/// PolyLog[0, z] = z/(1-z)
pub fn polylog_s0(z_expr: &Expr) -> Result<Expr, InterpreterError> {
  match z_expr {
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    Expr::Integer(1) => Ok(Expr::Identifier("ComplexInfinity".to_string())),
    Expr::Integer(z) => {
      // z/(1-z) as rational
      Ok(make_rational(*z, 1 - z))
    }
    Expr::Real(f) => Ok(Expr::Real(f / (1.0 - f))),
    _ => {
      // z/(1-z)
      Ok(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(z_expr.clone()),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(Expr::Integer(1)),
          right: Box::new(z_expr.clone()),
        }),
      })
    }
  }
}

/// PolyLog[-n, z] for n >= 1 using Eulerian numbers
pub fn polylog_negative_s(
  n: usize,
  z_expr: &Expr,
) -> Result<Expr, InterpreterError> {
  match z_expr {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Integer(1) => {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    Expr::Integer(z) => {
      // Evaluate numerically for integer z != 0, 1
      let zf = *z as f64;
      return Ok(Expr::Real(polylog_numeric(-(n as f64), zf)));
    }
    Expr::Real(f) => return Ok(Expr::Real(polylog_numeric(-(n as f64), *f))),
    _ => {}
  }

  // Compute Eulerian numbers A(n, k) for k = 0..n-1
  let eulerian = eulerian_numbers(n);

  // Build numerator: Σ A(n, k) * x^{k+1}
  let mut terms: Vec<Expr> = Vec::new();
  for (k, &a) in eulerian.iter().enumerate() {
    if a == 0 {
      continue;
    }
    let x_power = if k + 1 == 1 {
      z_expr.clone()
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(z_expr.clone()),
        right: Box::new(Expr::Integer((k + 1) as i128)),
      }
    };
    let term = if a == 1 {
      x_power
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(a)),
        right: Box::new(x_power),
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

  // Denominator: (1 - x)^{n+1}
  let one_minus_x = Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(Expr::Integer(1)),
    right: Box::new(z_expr.clone()),
  };
  let denominator = Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(one_minus_x),
    right: Box::new(Expr::Integer((n + 1) as i128)),
  };

  Ok(Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(numerator),
    right: Box::new(denominator),
  })
}

/// Compute Eulerian numbers A(n, k) for k = 0, ..., n-1
pub fn eulerian_numbers(n: usize) -> Vec<i128> {
  if n == 0 {
    return vec![];
  }
  let mut a = vec![0_i128; n];
  a[0] = 1;
  for m in 2..=n {
    let mut new_a = vec![0_i128; n];
    new_a[0] = 1;
    for k in 1..m {
      new_a[k] = (k as i128 + 1) * a[k] + (m as i128 - k as i128) * a[k - 1];
    }
    a = new_a;
  }
  a[..n].to_vec()
}

/// PolyLog[s, -1] = -(1 - 2^{1-s}) * Zeta[s] for s >= 2
pub fn polylog_at_neg1(s: i128) -> Result<Expr, InterpreterError> {
  let s_usize = s as usize;

  if s % 2 == 0 {
    // Even s: Zeta[s] is exact
    if let Some((b_num, b_den)) = bernoulli_number(s_usize) {
      if b_num == 0 {
        return Ok(Expr::Integer(0));
      }

      // Compute Zeta coefficient: |B_{2n}| * 2^{2n-1} / (2n)!
      let mut znum = b_num.abs();
      let mut zden = b_den.abs();
      for _ in 0..(s_usize - 1) {
        znum = match znum.checked_mul(2) {
          Some(v) => v,
          None => return Ok(unevaluated_polylog(s, -1)),
        };
        let g = gcd(znum, zden);
        znum /= g;
        zden /= g;
      }
      for k in 1..=s_usize {
        zden = match zden.checked_mul(k as i128) {
          Some(v) => v,
          None => return Ok(unevaluated_polylog(s, -1)),
        };
        let g = gcd(znum, zden);
        znum /= g;
        zden /= g;
      }

      // Multiply by -(1 - 2^{1-s}) = (1 - 2^{s-1}) / 2^{s-1}
      let pow2 = 1_i128 << (s_usize - 1);
      let coeff_num = 1 - pow2; // negative
      let coeff_den = pow2;

      let mut final_num = match coeff_num.checked_mul(znum) {
        Some(v) => v,
        None => return Ok(unevaluated_polylog(s, -1)),
      };
      let mut final_den = match coeff_den.checked_mul(zden) {
        Some(v) => v,
        None => return Ok(unevaluated_polylog(s, -1)),
      };
      if final_den < 0 {
        final_num = -final_num;
        final_den = -final_den;
      }
      let g = gcd(final_num.abs(), final_den);
      final_num /= g;
      final_den /= g;

      // Build coefficient * Pi^s
      let pi_power = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::Identifier("Pi".to_string())),
        right: Box::new(Expr::Integer(s)),
      };

      if final_num.abs() == 1 && final_den == 1 {
        if final_num == 1 {
          return Ok(pi_power);
        } else {
          return Ok(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(pi_power),
          });
        }
      } else if final_num.abs() == 1 {
        return Ok(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(make_rational(final_num, final_den)),
          right: Box::new(pi_power),
        });
      } else if final_den == 1 {
        return Ok(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(final_num)),
          right: Box::new(pi_power),
        });
      } else {
        return Ok(Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(final_num)),
            right: Box::new(pi_power),
          }),
          right: Box::new(Expr::Integer(final_den)),
        });
      }
    }
  } else {
    // Odd s >= 3: Zeta stays symbolic
    // PolyLog[s, -1] = cn/cd * Zeta[s] where cn < 0
    let pow2 = 1_i128 << (s_usize - 1);
    let coeff_num = 1 - pow2;
    let coeff_den = pow2;
    let g = gcd(coeff_num.abs(), coeff_den);
    let cn = coeff_num / g;
    let cd = coeff_den / g;

    let zeta_expr = Expr::FunctionCall {
      name: "Zeta".to_string(),
      args: vec![Expr::Integer(s)],
    };

    if cd == 1 {
      return Ok(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(cn)),
        right: Box::new(zeta_expr),
      });
    }
    // Use (cn*Zeta[s])/cd format
    return Ok(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(cn)),
        right: Box::new(zeta_expr),
      }),
      right: Box::new(Expr::Integer(cd)),
    });
  }

  Ok(unevaluated_polylog(s, -1))
}

pub fn unevaluated_polylog(s: i128, z: i128) -> Expr {
  Expr::FunctionCall {
    name: "PolyLog".to_string(),
    args: vec![Expr::Integer(s), Expr::Integer(z)],
  }
}

/// Compute polylogarithm numerically using series summation
pub fn polylog_numeric(s: f64, z: f64) -> f64 {
  if z == 0.0 {
    return 0.0;
  }
  if z.abs() <= 1.0 {
    let mut sum = 0.0;
    let mut z_power = z;
    for k in 1..=10000 {
      let term = z_power / (k as f64).powf(s);
      sum += term;
      if term.abs() < 1e-15 * sum.abs().max(1e-300) {
        break;
      }
      z_power *= z;
    }
    sum
  } else {
    f64::NAN
  }
}
