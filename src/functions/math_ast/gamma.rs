#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr};
use num_bigint::BigInt;

/// Pochhammer[a, n] - Rising factorial (Pochhammer symbol): a * (a+1) * ... * (a+n-1)
pub fn pochhammer_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Pochhammer expects exactly 2 arguments".into(),
    ));
  }
  // Pochhammer[a, 0] = 1 for any a
  if matches!(&args[1], Expr::Integer(0)) {
    return Ok(Expr::Integer(1));
  }
  // Both arguments are numeric integers → compute directly
  if let (Some(a), Some(n)) = (expr_to_i128(&args[0]), expr_to_i128(&args[1])) {
    if n < 0 {
      // Pochhammer[a, -n] = 1/((a-1)(a-2)...(a-n))
      let abs_n = (-n) as usize;
      let mut denom = BigInt::from(1);
      for i in 1..=abs_n as i128 {
        denom *= BigInt::from(a - i);
      }
      let denom_expr = bigint_to_expr(denom);
      let result = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(denom_expr),
      };
      return crate::evaluator::evaluate_expr_to_expr(&result);
    }
    let mut result = BigInt::from(1);
    for i in 0..n {
      result *= BigInt::from(a + i);
    }
    Ok(bigint_to_expr(result))
  } else if let Some(n) = expr_to_i128(&args[1]) {
    // n is a concrete integer, a is symbolic → expand symbolically
    let a_expr = &args[0];
    if n > 0 && n <= 20 {
      // Pochhammer[a, n] = a * (a+1) * ... * (a+n-1)
      let factors: Vec<Expr> = (0..n)
        .map(|i| {
          if i == 0 {
            a_expr.clone()
          } else {
            Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(Expr::Integer(i)),
              right: Box::new(a_expr.clone()),
            }
          }
        })
        .collect();
      let product = factors
        .into_iter()
        .reduce(|acc, f| Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(acc),
          right: Box::new(f),
        })
        .unwrap();
      crate::evaluator::evaluate_expr_to_expr(&product)
    } else if (-20..0).contains(&n) {
      // Pochhammer[a, -n] = 1/((a-1)(a-2)...(a-n))
      let abs_n = (-n) as usize;
      let factors: Vec<Expr> = (1..=abs_n)
        .map(|i| Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::Integer(-(i as i128))),
          right: Box::new(a_expr.clone()),
        })
        .collect();
      let denom = factors
        .into_iter()
        .reduce(|acc, f| Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(acc),
          right: Box::new(f),
        })
        .unwrap();
      let result = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(denom),
      };
      crate::evaluator::evaluate_expr_to_expr(&result)
    } else {
      Ok(Expr::FunctionCall {
        name: "Pochhammer".to_string(),
        args: args.to_vec(),
      })
    }
  } else {
    // Numeric evaluation: Pochhammer[a, n] = Gamma[a + n] / Gamma[a]
    if let (Some(a_f), Some(n_f)) =
      (try_eval_to_f64(&args[0]), try_eval_to_f64(&args[1]))
    {
      let has_real =
        matches!(&args[0], Expr::Real(_)) || matches!(&args[1], Expr::Real(_));
      if has_real {
        let gamma_a = super::gamma_fn(a_f);
        let gamma_a_n = super::gamma_fn(a_f + n_f);
        if gamma_a.is_finite()
          && gamma_a_n.is_finite()
          && gamma_a.abs() > 1e-300
        {
          return Ok(Expr::Real(gamma_a_n / gamma_a));
        }
      }
    }
    Ok(Expr::FunctionCall {
      name: "Pochhammer".to_string(),
      args: args.to_vec(),
    })
  }
}

/// FactorialPower[n, k] - falling factorial: n*(n-1)*...*(n-k+1)
/// FactorialPower[n, k, h] - generalized: n*(n-h)*(n-2h)*... (k terms)
pub fn factorial_power_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "FactorialPower expects 2 or 3 arguments".into(),
    ));
  }
  let h = if args.len() == 3 {
    expr_to_i128(&args[2])
  } else {
    Some(1)
  };
  if let (Some(n), Some(k), Some(h)) =
    (expr_to_i128(&args[0]), expr_to_i128(&args[1]), h)
  {
    if k == 0 {
      return Ok(Expr::Integer(1));
    }
    if k < 0 {
      return Ok(Expr::FunctionCall {
        name: "FactorialPower".to_string(),
        args: args.to_vec(),
      });
    }
    let mut result = BigInt::from(1);
    for i in 0..k {
      result *= BigInt::from(n - i * h);
    }
    Ok(bigint_to_expr(result))
  } else {
    Ok(Expr::FunctionCall {
      name: "FactorialPower".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Gamma[n] - Gamma function: Gamma[n] = (n-1)! for positive integers
pub fn gamma_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Gamma expects 1 or 2 arguments".into(),
    ));
  }

  // Two-argument form: Gamma[a, z] = upper incomplete gamma function
  if args.len() == 2 {
    return gamma_incomplete_upper(&args[0], &args[1]);
  }
  match expr_to_i128(&args[0]) {
    Some(n) => {
      if n <= 0 {
        // Gamma has poles at non-positive integers
        return Ok(Expr::Identifier("ComplexInfinity".to_string()));
      }
      // Gamma[n] = (n-1)! for positive integers
      let mut result = BigInt::from(1);
      for i in 2..n {
        result *= i;
      }
      Ok(bigint_to_expr(result))
    }
    None if matches!(&args[0], Expr::Real(_)) => {
      let f = if let Expr::Real(f) = &args[0] {
        *f
      } else {
        unreachable!()
      };
      if f <= 0.0 && f.fract() == 0.0 {
        // Poles at non-positive integers
        return Ok(Expr::Identifier("ComplexInfinity".to_string()));
      }
      // Use Stirling's approximation via the standard library's tgamma equivalent
      // Rust doesn't have tgamma in std, but we can compute via the Lanczos approximation
      let result = gamma_fn(f);
      if result.is_infinite() {
        Ok(Expr::Identifier("ComplexInfinity".to_string()))
      } else {
        Ok(Expr::Real(result))
      }
    }
    // Gamma[n/2] for odd n — half-integer values
    // Gamma[1/2] = Sqrt[Pi], Gamma[3/2] = Sqrt[Pi]/2,
    // Gamma[(2k+1)/2] = (2k)! * Sqrt[Pi] / (4^k * k!)
    // Gamma[(-2k+1)/2] uses reflection: Gamma[-n+1/2] = (-1)^n * Pi / (Gamma[n+1/2] * Sin(Pi*(n+1/2)))
    //   simplified: Gamma[1/2 - n] = (-4)^n * n! * Sqrt[Pi] / (2n)!
    _ if matches!(&args[0], Expr::FunctionCall { name, args: ra }
      if name == "Rational" && ra.len() == 2
        && matches!(&ra[1], Expr::Integer(2))
        && matches!(&ra[0], Expr::Integer(n) if n % 2 != 0)
    ) =>
    {
      if let Expr::FunctionCall { args: ra, .. } = &args[0]
        && let Expr::Integer(num) = &ra[0]
      {
        let num = *num;
        // num is odd, denominator is 2, so argument is num/2
        // For positive half-integers: Gamma[(2k+1)/2] where k = (num-1)/2
        if num > 0 {
          let k = (num - 1) / 2;
          // Gamma[(2k+1)/2] = (2k)! * Sqrt[Pi] / (4^k * k!)
          let mut factorial_2k = BigInt::from(1);
          for i in 2..=(2 * k) {
            factorial_2k *= i;
          }
          let mut factorial_k = BigInt::from(1);
          for i in 2..=k {
            factorial_k *= i;
          }
          let four_k = BigInt::from(4).pow(k as u32);
          let denom = four_k * factorial_k;
          // Result = factorial_2k / denom * Sqrt[Pi]
          // Simplify the rational part
          let g = gcd_bigint(&factorial_2k, &denom);
          let num_simplified = &factorial_2k / &g;
          let den_simplified = &denom / &g;
          let sqrt_pi = Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![
              Expr::Identifier("Pi".to_string()),
              Expr::FunctionCall {
                name: "Rational".to_string(),
                args: vec![Expr::Integer(1), Expr::Integer(2)],
              },
            ],
          };
          if den_simplified == BigInt::from(1) {
            if num_simplified == BigInt::from(1) {
              return Ok(sqrt_pi);
            }
            return Ok(Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![bigint_to_expr(num_simplified), sqrt_pi],
            });
          }
          let coeff = Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![
              bigint_to_expr(num_simplified),
              bigint_to_expr(den_simplified),
            ],
          };
          return Ok(Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![coeff, sqrt_pi],
          });
        } else if num < 0 {
          // Negative half-integers: Gamma[(1-2n)/2] = Gamma[1/2 - n]
          // = (-4)^n * n! * Sqrt[Pi] / (2n)!  where n = (1 - num) / 2
          let n = (1 - num) / 2;
          let sign = if n % 2 == 0 {
            BigInt::from(1)
          } else {
            BigInt::from(-1)
          };
          let four_n = BigInt::from(4).pow(n as u32);
          let mut factorial_n = BigInt::from(1);
          for i in 2..=n {
            factorial_n *= i;
          }
          let mut factorial_2n = BigInt::from(1);
          for i in 2..=(2 * n) {
            factorial_2n *= i;
          }
          let numerator = &sign * &four_n * &factorial_n;
          let denominator = factorial_2n;
          let num_abs: BigInt = numerator.magnitude().clone().into();
          let g = gcd_bigint(&num_abs, &denominator);
          let num_simplified = &num_abs / &g;
          let den_simplified = &denominator / &g;
          let is_neg = numerator < BigInt::from(0);
          let sqrt_pi = Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![
              Expr::Identifier("Pi".to_string()),
              Expr::FunctionCall {
                name: "Rational".to_string(),
                args: vec![Expr::Integer(1), Expr::Integer(2)],
              },
            ],
          };
          let coeff_num = if is_neg {
            -num_simplified.clone()
          } else {
            num_simplified.clone()
          };
          if den_simplified == BigInt::from(1) {
            if coeff_num == BigInt::from(1) {
              return Ok(sqrt_pi);
            }
            return Ok(Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![bigint_to_expr(coeff_num), sqrt_pi],
            });
          }
          let coeff = Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![
              bigint_to_expr(coeff_num),
              bigint_to_expr(den_simplified),
            ],
          };
          return Ok(Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![coeff, sqrt_pi],
          });
        }
      }
      Ok(Expr::FunctionCall {
        name: "Gamma".to_string(),
        args: args.to_vec(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "Gamma".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Upper incomplete gamma function Gamma[a, z]
fn gamma_incomplete_upper(
  a: &Expr,
  z: &Expr,
) -> Result<Expr, InterpreterError> {
  // Special case: Gamma[1, z] = E^(-z)
  if matches!(a, Expr::Integer(1)) {
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![
        Expr::Identifier("E".to_string()),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), z.clone()],
        },
      ],
    });
  }

  // Special case: Gamma[0, z] = ExpIntegralE[1, z]
  if matches!(a, Expr::Integer(0)) {
    return Ok(Expr::FunctionCall {
      name: "ExpIntegralE".to_string(),
      args: vec![Expr::Integer(1), z.clone()],
    });
  }

  // Numeric evaluation: both args are real numbers
  if let (Some(a_val), Some(z_val)) = (try_eval_to_f64(a), try_eval_to_f64(z))
    && z_val >= 0.0
  {
    let result = upper_incomplete_gamma(a_val, z_val);
    if result.is_finite() {
      return Ok(Expr::Real(result));
    }
  }

  // For positive integer a: Gamma[n, z] = (n-1)! * E^(-z) * Sum[z^k/k!, {k, 0, n-1}]
  if let Some(n) = expr_to_i128(a)
    && n > 0
  {
    let n = n as usize;
    // Build the sum: sum_{k=0}^{n-1} z^k / k!
    let mut terms = Vec::new();
    let mut factorial: i128 = 1;
    for k in 0..n {
      if k > 0 {
        factorial *= k as i128;
      }
      let z_power = if k == 0 {
        Expr::Integer(1)
      } else if k == 1 {
        z.clone()
      } else {
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![z.clone(), Expr::Integer(k as i128)],
        }
      };
      let term = if factorial == 1 {
        z_power
      } else {
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(1), Expr::Integer(factorial)],
            },
            z_power,
          ],
        }
      };
      terms.push(term);
    }
    let sum = if terms.len() == 1 {
      terms.into_iter().next().unwrap()
    } else {
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: terms,
      }
    };
    // (n-1)! * E^(-z) * sum
    let mut n_minus_1_factorial: i128 = 1;
    for i in 2..n as i128 {
      n_minus_1_factorial *= i;
    }
    let exp_neg_z = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![
        Expr::Identifier("E".to_string()),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), z.clone()],
        },
      ],
    };
    let result = if n_minus_1_factorial == 1 {
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![exp_neg_z, sum],
      }
    } else {
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(n_minus_1_factorial), exp_neg_z, sum],
      }
    };
    return crate::evaluator::evaluate_expr_to_expr(&result);
  }

  // Default: return unevaluated
  Ok(Expr::FunctionCall {
    name: "Gamma".to_string(),
    args: vec![a.clone(), z.clone()],
  })
}

/// Numerical upper incomplete gamma via continued fraction (Legendre)
fn upper_incomplete_gamma(a: f64, z: f64) -> f64 {
  if z == 0.0 {
    return gamma_fn(a);
  }
  // Use series for small z, continued fraction for large z
  if z < a + 1.0 {
    // Gamma(a, z) = Gamma(a) - gamma_lower(a, z)
    gamma_fn(a) - lower_incomplete_gamma_series(a, z)
  } else {
    // Continued fraction representation (Legendre)
    upper_incomplete_gamma_cf(a, z)
  }
}

/// Lower incomplete gamma via series expansion
fn lower_incomplete_gamma_series(a: f64, z: f64) -> f64 {
  let mut sum = 1.0 / a;
  let mut term = 1.0 / a;
  for n in 1..200 {
    term *= z / (a + n as f64);
    sum += term;
    if term.abs() < 1e-15 * sum.abs() {
      break;
    }
  }
  sum * (-z).exp() * z.powf(a)
}

/// Upper incomplete gamma via continued fraction
fn upper_incomplete_gamma_cf(a: f64, z: f64) -> f64 {
  // Modified Lentz's method
  let mut c = 1e-30_f64;
  let mut d = z + 1.0 - a;
  if d.abs() < 1e-30 {
    d = 1e-30;
  }
  d = 1.0 / d;
  let mut f = d;
  for n in 1..200 {
    let an = n as f64 * (a - n as f64);
    let bn = z + (2 * n + 1) as f64 - a;
    d = bn + an * d;
    if d.abs() < 1e-30 {
      d = 1e-30;
    }
    c = bn + an / c;
    if c.abs() < 1e-30 {
      c = 1e-30;
    }
    d = 1.0 / d;
    let delta = c * d;
    f *= delta;
    if (delta - 1.0).abs() < 1e-15 {
      break;
    }
  }
  f * (-z).exp() * z.powf(a)
}

fn gcd_bigint(a: &BigInt, b: &BigInt) -> BigInt {
  use num_traits::Zero;
  let mut a = a.magnitude().clone();
  let mut b = b.magnitude().clone();
  while !b.is_zero() {
    let t = b.clone();
    b = &a % &b;
    a = t;
  }
  a.into()
}

/// Lanczos approximation for the Gamma function
pub fn gamma_fn(x: f64) -> f64 {
  if x < 0.5 {
    // Reflection formula: Gamma(1-z) * Gamma(z) = pi / sin(pi*z)
    std::f64::consts::PI
      / ((std::f64::consts::PI * x).sin() * gamma_fn(1.0 - x))
  } else {
    let x = x - 1.0;
    let g = 7.0;
    let c = [
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
    let mut sum = c[0];
    for (i, &ci) in c.iter().enumerate().skip(1) {
      sum += ci / (x + i as f64);
    }
    let t = x + g + 0.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * sum
  }
}

/// Beta[a, b] - Euler beta function
/// Beta[a, b] = Gamma[a] * Gamma[b] / Gamma[a + b]
pub fn beta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Beta expects exactly 2 arguments".into(),
    ));
  }

  // Try to evaluate for positive integer arguments
  // Beta[m, n] = (m-1)! * (n-1)! / (m+n-1)!
  if let (Expr::Integer(a), Expr::Integer(b)) = (&args[0], &args[1])
    && *a > 0
    && *b > 0
  {
    let a_u = (*a - 1) as usize;
    let b_u = (*b - 1) as usize;
    let ab_u = (*a + *b - 1) as usize;
    if let (Some(a_fact), Some(b_fact), Some(ab_fact)) = (
      factorial_i128(a_u),
      factorial_i128(b_u),
      factorial_i128(ab_u),
    ) {
      return Ok(make_rational(a_fact * b_fact, ab_fact));
    }
  }

  // Try rational args for half-integer cases
  // Beta[p/q, r/s] for half-integers involves Gamma at half-integers
  if let (Some(a_f), Some(b_f)) = (expr_to_f64(&args[0]), expr_to_f64(&args[1]))
  {
    // Check if both are positive half-integers (n + 1/2)
    let a_half = a_f * 2.0;
    let b_half = b_f * 2.0;
    if a_half == a_half.round()
      && b_half == b_half.round()
      && a_f > 0.0
      && b_f > 0.0
      && a_half.fract() == 0.0
      && b_half.fract() == 0.0
    {
      let a2 = a_half as i128;
      let b2 = b_half as i128;

      // At least one must be odd (half-integer) for Pi to appear
      if a2 % 2 != 0 || b2 % 2 != 0 {
        // Check if both args are Real (then numeric)
        if matches!(&args[0], Expr::Real(_))
          || matches!(&args[1], Expr::Real(_))
        {
          let result = gamma_fn(a_f) * gamma_fn(b_f) / gamma_fn(a_f + b_f);
          return Ok(Expr::Real(result));
        }
        // For exact half-integer arguments, compute via Gamma
        // If sum is integer, result is rational * sqrt(pi) or rational * pi
        let sum2 = a2 + b2;
        if sum2 % 2 == 0 {
          // Both half-integers or both integers, sum is integer
          // Result involves sqrt(pi) terms that may cancel
          // Use numeric for now unless both are half-integers with integer sum
          // Beta[a, b] = Γ(a)Γ(b)/Γ(a+b)
          // When a, b are half-integers, Γ(n+1/2) = (2n)! sqrt(π) / (4^n n!)
          // So Gamma product has π, and if sum is integer, Γ(sum) is (sum-1)!
          // Beta = Γ(a)Γ(b) / (sum-1)!
          let sum_int = (sum2 / 2) as usize;
          if let Some(sum_fact) = factorial_i128(sum_int - 1) {
            // Compute Γ(a) * Γ(b) / (sum-1)! where a, b are half-integers
            // Γ(k/2) for odd k: Γ((2m+1)/2) = (2m)! π^{1/2} / (4^m m!)
            // For even k: Γ(k/2) = ((k/2)-1)!
            let gamma_a = gamma_half_integer_parts(a2);
            let gamma_b = gamma_half_integer_parts(b2);
            if let (
              Some((a_num, a_den, a_pi_pow)),
              Some((b_num, b_den, b_pi_pow)),
            ) = (gamma_a, gamma_b)
            {
              let total_pi_pow = a_pi_pow + b_pi_pow; // each half-integer contributes 1/2
              let num =
                a_num.checked_mul(b_num).unwrap_or_else(|| a_num * b_num);
              let den = a_den
                .checked_mul(b_den)
                .and_then(|v| v.checked_mul(sum_fact))
                .unwrap_or(1);
              let g = gcd(num.abs(), den.abs());
              let (num, den) = if g > 0 {
                (num / g, den / g)
              } else {
                (num, den)
              };

              if total_pi_pow == 0 {
                return Ok(make_rational(num, den));
              } else if total_pi_pow == 2 {
                // Two sqrt(Pi) factors = Pi
                // Result is (num/den) * Pi
                if den == 1 {
                  if num == 1 {
                    return Ok(Expr::Identifier("Pi".to_string()));
                  }
                  return Ok(Expr::BinaryOp {
                    op: BinaryOperator::Times,
                    left: Box::new(Expr::Integer(num)),
                    right: Box::new(Expr::Identifier("Pi".to_string())),
                  });
                }
                return Ok(Expr::BinaryOp {
                  op: BinaryOperator::Divide,
                  left: Box::new(if num == 1 {
                    Expr::Identifier("Pi".to_string())
                  } else {
                    Expr::BinaryOp {
                      op: BinaryOperator::Times,
                      left: Box::new(Expr::Integer(num)),
                      right: Box::new(Expr::Identifier("Pi".to_string())),
                    }
                  }),
                  right: Box::new(Expr::Integer(den)),
                });
              }
            }
          }
        }
      }
    }
  }

  // Numeric evaluation
  if let (Some(a_f), Some(b_f)) = (expr_to_f64(&args[0]), expr_to_f64(&args[1]))
    && (matches!(&args[0], Expr::Real(_)) || matches!(&args[1], Expr::Real(_)))
  {
    let result = gamma_fn(a_f) * gamma_fn(b_f) / gamma_fn(a_f + b_f);
    return Ok(Expr::Real(result));
  }

  // Unevaluated
  Ok(Expr::FunctionCall {
    name: "Beta".to_string(),
    args: args.to_vec(),
  })
}

/// LogGamma[z] — logarithm of the gamma function.
pub fn log_gamma_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "LogGamma".to_string(),
      args: args.to_vec(),
    });
  }

  let z = &args[0];

  // Handle exact integer cases
  if let Some(n) = expr_to_i128(z) {
    if n <= 0 {
      // LogGamma[0] = LogGamma[-n] = Infinity
      return Ok(Expr::Identifier("Infinity".to_string()));
    }
    if n == 1 || n == 2 {
      return Ok(Expr::Integer(0)); // Log[0!] = Log[1!] = 0
    }
    // LogGamma[n] = Log[(n-1)!]
    let gamma_result = gamma_ast(&[z.clone()])?;
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Log".to_string(),
      args: vec![gamma_result],
    });
  }

  // Handle Rational arguments — compute Gamma then Log
  if let Expr::FunctionCall { name, args: fargs } = z
    && name == "Rational"
    && fargs.len() == 2
    && let (Expr::Integer(n), Expr::Integer(d)) = (&fargs[0], &fargs[1])
  {
    if *d == 2 && *n > 0 {
      // Half-integer: LogGamma[k/2] = Log[Gamma[k/2]]
      let gamma_result = gamma_ast(&[z.clone()])?;
      return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![gamma_result],
      });
    }
    if *n <= 0 && *d > 0 && *n % *d == 0 {
      // Non-positive integer
      return Ok(Expr::Identifier("Infinity".to_string()));
    }
  }

  // Handle numeric (Real) — use lgamma
  if let Some(f) = try_eval_to_f64(z)
    && (matches!(z, Expr::Real(_))
      || matches!(z, Expr::FunctionCall { name, .. } if name == "Rational")
        && try_eval_to_f64(z).is_some())
  {
    if f <= 0.0 && f == f.floor() {
      return Ok(Expr::Identifier("Infinity".to_string()));
    }
    if matches!(z, Expr::Real(_)) {
      let result = gamma_fn(f).abs().ln();
      return Ok(Expr::Real(result));
    }
  }

  // Return unevaluated for symbolic case
  Ok(Expr::FunctionCall {
    name: "LogGamma".to_string(),
    args: args.to_vec(),
  })
}

/// Compute parts of Gamma at half-integer: Gamma(k/2) for integer k > 0
/// Returns (numerator, denominator, pi_power) where result = (num/den) * Pi^(pi_power/2)
/// pi_power is 0 or 1 (representing sqrt(Pi)^pi_power)
pub fn gamma_half_integer_parts(k2: i128) -> Option<(i128, i128, i128)> {
  if k2 <= 0 {
    return None;
  }
  if k2 % 2 == 0 {
    // k2 = 2m, so Gamma(m) = (m-1)!
    let m = (k2 / 2) as usize;
    let fact = factorial_i128(m - 1)?;
    Some((fact, 1, 0))
  } else {
    // k2 = 2m+1, so Gamma(m + 1/2) = (2m)! * sqrt(pi) / (4^m * m!)
    let m = ((k2 - 1) / 2) as usize;
    let two_m_fact = factorial_i128(2 * m)?;
    let m_fact = factorial_i128(m)?;
    let four_m = 4i128.checked_pow(m as u32)?;
    Some((two_m_fact, four_m * m_fact, 1))
  }
}

/// BetaRegularized[z, a, b] - Regularized incomplete beta function I_z(a, b)
/// I_z(a, b) = B(z; a, b) / B(a, b) where B(z; a, b) = ∫₀ᶻ t^(a-1) (1-t)^(b-1) dt
pub fn beta_regularized_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "BetaRegularized expects exactly 3 arguments".into(),
    ));
  }

  let z_expr = &args[0];
  let a_expr = &args[1];
  let b_expr = &args[2];

  // BetaRegularized[0, a, b] = 0 (when a > 0)
  if is_expr_zero(z_expr) && is_positive_numeric(a_expr) {
    return Ok(Expr::Integer(0));
  }

  // BetaRegularized[1, a, b] = 1 (when b > 0)
  if matches!(z_expr, Expr::Integer(1)) && is_positive_numeric(b_expr) {
    return Ok(Expr::Integer(1));
  }

  // Numeric evaluation when all arguments are numeric and at least one is Real
  let z_val = expr_to_f64(z_expr);
  let a_val = expr_to_f64(a_expr);
  let b_val = expr_to_f64(b_expr);
  let has_real = matches!(z_expr, Expr::Real(_))
    || matches!(a_expr, Expr::Real(_))
    || matches!(b_expr, Expr::Real(_));

  if let (Some(z), Some(a), Some(b)) = (z_val, a_val, b_val)
    && has_real
  {
    return Ok(Expr::Real(beta_regularized_numeric(z, a, b)));
  }

  // Unevaluated
  Ok(Expr::FunctionCall {
    name: "BetaRegularized".to_string(),
    args: args.to_vec(),
  })
}

/// Check if an expression is a positive number
fn is_positive_numeric(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(n) => *n > 0,
    Expr::Real(x) => *x > 0.0,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        (*n > 0 && *d > 0) || (*n < 0 && *d < 0)
      } else {
        false
      }
    }
    _ => false,
  }
}

/// Compute the regularized incomplete beta function I_x(a, b) numerically
/// Uses the continued fraction representation (Lentz's algorithm)
fn beta_regularized_numeric(x: f64, a: f64, b: f64) -> f64 {
  if x <= 0.0 {
    return 0.0;
  }
  if x >= 1.0 {
    return 1.0;
  }

  // Use symmetry relation: I_x(a,b) = 1 - I_{1-x}(b,a) when x > (a+1)/(a+b+2)
  // This ensures convergence of the continued fraction
  if x > (a + 1.0) / (a + b + 2.0) {
    return 1.0 - beta_regularized_numeric(1.0 - x, b, a);
  }

  // Compute using the continued fraction representation
  // I_x(a,b) = x^a * (1-x)^b / (a * B(a,b)) * 1/(1+ d1/(1+ d2/(1+ ...)))
  // where d_{2m+1} = -(a+m)(a+b+m) x / ((a+2m)(a+2m+1))
  //       d_{2m}   = m(b-m) x / ((a+2m-1)(a+2m))

  let ln_prefix = a * x.ln() + b * (1.0 - x).ln() - a.ln() - ln_beta(a, b);
  let prefix = ln_prefix.exp();

  // Lentz's continued fraction method
  let mut c = 1.0_f64;
  let mut d = 1.0 - (a + b) * x / (a + 1.0);
  if d.abs() < 1e-30 {
    d = 1e-30;
  }
  d = 1.0 / d;
  let mut result = d;

  for m in 1..200 {
    let m_f = m as f64;

    // Even step: d_{2m}
    let numerator =
      m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
    d = 1.0 + numerator * d;
    if d.abs() < 1e-30 {
      d = 1e-30;
    }
    c = 1.0 + numerator / c;
    if c.abs() < 1e-30 {
      c = 1e-30;
    }
    d = 1.0 / d;
    result *= d * c;

    // Odd step: d_{2m+1}
    let numerator = -(a + m_f) * (a + b + m_f) * x
      / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
    d = 1.0 + numerator * d;
    if d.abs() < 1e-30 {
      d = 1e-30;
    }
    c = 1.0 + numerator / c;
    if c.abs() < 1e-30 {
      c = 1e-30;
    }
    d = 1.0 / d;
    let delta = d * c;
    result *= delta;

    if (delta - 1.0).abs() < 1e-15 {
      break;
    }
  }

  prefix * result
}

/// Compute ln(Beta(a, b)) = ln(Gamma(a)) + ln(Gamma(b)) - ln(Gamma(a+b))
fn ln_beta(a: f64, b: f64) -> f64 {
  lgamma(a) + lgamma(b) - lgamma(a + b)
}

/// Log-gamma function using Lanczos approximation
pub fn lgamma(x: f64) -> f64 {
  // Use the standard library's ln_gamma if available via f64 methods
  // Otherwise use Lanczos approximation
  let g = 7.0;
  let coefs = [
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

  if x < 0.5 {
    // Reflection formula: Gamma(1-x) * Gamma(x) = pi / sin(pi*x)
    let log_pi = std::f64::consts::PI.ln();
    log_pi - (std::f64::consts::PI * x).sin().abs().ln() - lgamma(1.0 - x)
  } else {
    let x = x - 1.0;
    let mut base = coefs[0];
    for (i, &c) in coefs.iter().enumerate().skip(1) {
      base += c / (x + i as f64);
    }
    let t = x + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (t.ln() * (x + 0.5)) - t
      + base.ln()
  }
}

/// GammaRegularized[a, z] - Regularized upper incomplete gamma function Q(a, z)
/// Q(a, z) = Gamma(a, z) / Gamma(a) = 1 - P(a, z)
pub fn gamma_regularized_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "GammaRegularized expects exactly 2 arguments".into(),
    ));
  }

  let a_expr = &args[0];
  let z_expr = &args[1];

  // GammaRegularized[a, 0] = 1
  if is_expr_zero(z_expr) && is_positive_numeric(a_expr) {
    return Ok(Expr::Integer(1));
  }

  // GammaRegularized[a, Infinity] = 0
  if matches!(z_expr, Expr::Identifier(s) if s == "Infinity") {
    return Ok(Expr::Integer(0));
  }

  // Numeric evaluation
  let a_val = expr_to_f64(a_expr);
  let z_val = expr_to_f64(z_expr);
  let has_real =
    matches!(a_expr, Expr::Real(_)) || matches!(z_expr, Expr::Real(_));

  if let (Some(a), Some(z)) = (a_val, z_val)
    && has_real
  {
    return Ok(Expr::Real(gamma_regularized_numeric(a, z)));
  }

  // Unevaluated
  Ok(Expr::FunctionCall {
    name: "GammaRegularized".to_string(),
    args: args.to_vec(),
  })
}

/// Compute Q(a, z) = 1 - P(a, z) numerically
fn gamma_regularized_numeric(a: f64, z: f64) -> f64 {
  if z <= 0.0 {
    return 1.0;
  }

  if z < a + 1.0 {
    // Series expansion for P(a, z)
    let ln_prefix = a * z.ln() - z - lgamma(a);
    let prefix = ln_prefix.exp();

    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;
    for n in 1..200 {
      term *= z / (a + n as f64);
      sum += term;
      if term.abs() < 1e-15 * sum.abs() {
        break;
      }
    }
    1.0 - prefix * sum
  } else {
    // Continued fraction for Q(a, z) using Lentz's method
    let ln_prefix = a * z.ln() - z - lgamma(a);
    let prefix = ln_prefix.exp();

    let b0 = z - a + 1.0;
    let mut f = if b0.abs() < 1e-30 { 1e-30 } else { b0 };
    let mut c = f;
    let mut d;

    for n in 1..200 {
      let an = (n as f64) * (a - n as f64);
      let bn = z - a + 2.0 * n as f64 + 1.0;

      d = bn + an / f;
      if d.abs() < 1e-30 {
        d = 1e-30;
      }
      c = bn + an / c;
      if c.abs() < 1e-30 {
        c = 1e-30;
      }
      let delta = c / d;
      f *= delta;
      if (delta - 1.0).abs() < 1e-15 {
        break;
      }
    }

    prefix / f
  }
}

/// BarnesG[z] - Barnes G-function.
/// For positive integers n: G(n) = product of factorials = prod_{k=0}^{n-2} k!
/// G(1) = 1, G(n+1) = Gamma(n) * G(n)
/// For non-positive integers: G(n) = 0
/// For real values: numerical computation via log Barnes G
pub fn barnes_g_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "BarnesG expects 1 argument".into(),
    ));
  }

  match &args[0] {
    Expr::Integer(n) => {
      let n = *n;
      if n <= 0 {
        return Ok(Expr::Integer(0));
      }
      if n == 1 {
        return Ok(Expr::Integer(1));
      }
      // G(n) = product_{k=0}^{n-2} k!
      // Use recurrence: G(n+1) = Gamma(n) * G(n) = (n-1)! * G(n)
      // G(1) = 1, G(2) = 1, G(3) = 1, G(4) = 2, G(5) = 12, ...
      let mut result = num_bigint::BigInt::from(1);
      let mut factorial = num_bigint::BigInt::from(1);
      // factorial tracks (k-1)! as we compute G(k+1) = (k-1)! * G(k)
      for k in 1..n - 1 {
        factorial *= num_bigint::BigInt::from(k);
        result *= &factorial;
      }
      // Try to convert to i128
      use num_traits::ToPrimitive;
      match result.to_i128() {
        Some(v) => Ok(Expr::Integer(v)),
        None => Ok(Expr::BigInteger(result)),
      }
    }
    Expr::Real(x) => {
      let val = barnes_g_float(*x);
      Ok(Expr::Real(val))
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "Rational" && fargs.len() == 2 =>
    {
      // For N[BarnesG[rational]], return unevaluated; N[] will handle conversion
      Ok(Expr::FunctionCall {
        name: "BarnesG".to_string(),
        args: args.to_vec(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "BarnesG".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Compute Barnes G function for real z numerically.
/// Shifts z to [1, 2] via recurrence, then uses the Taylor series
/// derived from the Weierstrass product.
fn barnes_g_float(z: f64) -> f64 {
  if z == f64::INFINITY {
    return f64::INFINITY;
  }
  if z.is_nan() || z == f64::NEG_INFINITY {
    return f64::NAN;
  }

  // For non-positive integers, return 0
  if z <= 0.0 && z == z.floor() {
    return 0.0;
  }

  // Shift z to [1, 2] using recurrence G(z+1) = Gamma(z) * G(z)
  let mut z_shifted = z;
  let mut log_correction = 0.0_f64;

  while z_shifted < 1.0 {
    log_correction -= lgamma(z_shifted);
    z_shifted += 1.0;
  }
  while z_shifted > 1.5 {
    z_shifted -= 1.0;
    log_correction += lgamma(z_shifted);
  }

  let w = z_shifted - 1.0; // w in [0, 1]
  let log_g = log_barnes_g_series(w);

  (log_g + log_correction).exp()
}

/// Taylor series for ln G(1+z), valid for |z| ≤ 1.
/// Derived from the Weierstrass product:
/// ln G(1+z) = z/2 * ln(2π) - [z + (1+γ)z²]/2
///             + Σ_{m=2}^{∞} (-1)^m ζ(m)/(m+1) * z^{m+1}
fn log_barnes_g_series(z: f64) -> f64 {
  if z.abs() < 1e-15 {
    return 0.0;
  }

  let log_2pi = (2.0 * std::f64::consts::PI).ln();
  let gamma_e = 0.5772156649015329;

  let mut result = z / 2.0 * log_2pi - (z + (1.0 + gamma_e) * z * z) / 2.0;

  // Precomputed ζ(m) values for m=2..40
  let zeta: [f64; 39] = [
    1.6449340668482264,    // ζ(2)
    1.2020569031595943,    // ζ(3)
    1.0823232337111382,    // ζ(4)
    1.036_927_755_143_37,  // ζ(5)
    1.017_343_061_984_449, // ζ(6)
    1.0083492773819228,    // ζ(7)
    1.0040773561979443,    // ζ(8)
    1.0020083928260822,    // ζ(9)
    1.0009945751278181,    // ζ(10)
    1.0004941886041195,    // ζ(11)
    1.000_246_086_553_308, // ζ(12)
    1.0001227133475785,    // ζ(13)
    1.0000612481350587,    // ζ(14)
    1.000_030_588_236_307, // ζ(15)
    1.0000152822594086,    // ζ(16)
    1.0000076371976379,    // ζ(17)
    1.000_003_817_293_265, // ζ(18)
    1.0000019082127165,    // ζ(19)
    1.0000009539620339,    // ζ(20)
    1.0000004769329868,    // ζ(21)
    1.0000002384505027,    // ζ(22)
    1.000_000_119_219_926, // ζ(23)
    1.000_000_059_608_189, // ζ(24)
    1.0000000298035035,    // ζ(25)
    1.0000000149015548,    // ζ(26)
    1.0000000074507118,    // ζ(27)
    1.000_000_003_725_334, // ζ(28)
    1.0000000018626598,    // ζ(29)
    1.0000000009313274,    // ζ(30)
    1.0000000004656629,    // ζ(31)
    1.0000000002328312,    // ζ(32)
    1.0000000001164155,    // ζ(33)
    1.0000000000582077,    // ζ(34)
    1.0000000000291039,    // ζ(35)
    1.000_000_000_014_552, // ζ(36)
    1.000_000_000_007_276, // ζ(37)
    1.000_000_000_003_638, // ζ(38)
    1.000_000_000_001_819, // ζ(39)
    1.0000000000009095,    // ζ(40)
  ];

  let mut z_power = z * z * z; // z^3 (m=2 gives z^{m+1} = z^3)
  for (i, &zeta_m) in zeta.iter().enumerate() {
    let m = (i + 2) as f64;
    let sign = if (i + 2) % 2 == 0 { 1.0 } else { -1.0 };
    let term = sign * zeta_m / (m + 1.0) * z_power;
    result += term;
    if term.abs() < 1e-17 {
      break;
    }
    z_power *= z;
  }

  result
}
