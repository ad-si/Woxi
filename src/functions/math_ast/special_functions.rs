#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator};
use num_bigint::BigInt;

/// Pochhammer[a, n] - Rising factorial (Pochhammer symbol): a * (a+1) * ... * (a+n-1)
pub fn pochhammer_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Pochhammer expects exactly 2 arguments".into(),
    ));
  }
  if let (Some(a), Some(n)) = (expr_to_i128(&args[0]), expr_to_i128(&args[1])) {
    if n == 0 {
      return Ok(Expr::Integer(1));
    }
    if n < 0 {
      // Pochhammer[a, -n] = 1/((a-1)(a-2)...(a-n))
      return Ok(Expr::FunctionCall {
        name: "Pochhammer".to_string(),
        args: args.to_vec(),
      });
    }
    let mut result = BigInt::from(1);
    for i in 0..n {
      result *= BigInt::from(a + i);
    }
    Ok(bigint_to_expr(result))
  } else {
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
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Gamma expects exactly 1 argument".into(),
    ));
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

/// BesselJ[n, z] - Bessel function of the first kind
pub fn bessel_j_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "BesselJ expects exactly 2 arguments".into(),
    ));
  }
  let n_expr = &args[0];
  let z_expr = &args[1];

  // Extract numeric values
  let n_val = match n_expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  };
  let z_val = match z_expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  };

  // Special case: BesselJ[n, 0] for integer n
  if matches!(z_expr, Expr::Integer(0))
    && let Expr::Integer(n) = n_expr
  {
    return if *n == 0 {
      Ok(Expr::Integer(1))
    } else {
      Ok(Expr::Integer(0))
    };
  }

  // Numeric evaluation when both args are numeric and at least one is Real
  let is_numeric_eval = n_val.is_some()
    && z_val.is_some()
    && (matches!(z_expr, Expr::Real(_)) || matches!(n_expr, Expr::Real(_)));

  if is_numeric_eval {
    let n = n_val.unwrap();
    let z = z_val.unwrap();
    let result = bessel_j(n, z);
    return Ok(Expr::Real(result));
  }

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "BesselJ".to_string(),
    args: args.to_vec(),
  })
}

/// BesselJZero[n, k] — k-th positive zero of the Bessel function J_n
pub fn bessel_j_zero_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "BesselJZero expects exactly 2 arguments".into(),
    ));
  }

  // Extract numeric values
  let n_val = match &args[0] {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  };
  let k_val = match &args[1] {
    Expr::Integer(k) => Some(*k),
    Expr::Real(f) if *f == f.floor() && *f > 0.0 => Some(*f as i128),
    _ => None,
  };

  // Numeric evaluation when n is Real or k is Real
  let has_real =
    matches!(&args[0], Expr::Real(_)) || matches!(&args[1], Expr::Real(_));
  if has_real
    && let (Some(n), Some(k)) = (n_val, k_val)
    && k >= 1
  {
    let result = bessel_j_zero(n, k as usize);
    return Ok(Expr::Real(result));
  }

  // Return unevaluated (symbolic)
  Ok(Expr::FunctionCall {
    name: "BesselJZero".to_string(),
    args: args.to_vec(),
  })
}

/// Find the k-th positive zero of J_n(x) using bisection + Newton's method
fn bessel_j_zero(n: f64, k: usize) -> f64 {
  // Find zeros by scanning for sign changes, then refining with bisection
  let step = 0.5;
  let start = if n > 0.0 { n * 0.5 } else { step };
  let mut x = start;
  let mut prev_val = bessel_j(n, x);
  let mut zeros_found = 0;

  loop {
    let next_x = x + step;
    let next_val = bessel_j(n, next_x);

    if prev_val * next_val < 0.0 {
      // Sign change: there's a zero in [x, next_x]
      zeros_found += 1;
      if zeros_found == k {
        // Refine using bisection then Newton
        let mut lo = x;
        let mut hi = next_x;

        // Bisection to get close
        for _ in 0..60 {
          let mid = (lo + hi) / 2.0;
          let mid_val = bessel_j(n, mid);
          if mid_val == 0.0 {
            return mid;
          }
          if bessel_j(n, lo) * mid_val < 0.0 {
            hi = mid;
          } else {
            lo = mid;
          }
        }

        // Newton's method to polish
        let mut root = (lo + hi) / 2.0;
        for _ in 0..20 {
          let jn = bessel_j(n, root);
          // J'_n(x) = n/x * J_n(x) - J_{n+1}(x)
          let jn1 = bessel_j(n + 1.0, root);
          let deriv = (n / root) * jn - jn1;
          if deriv.abs() < 1e-300 {
            break;
          }
          let delta = jn / deriv;
          root -= delta;
          if delta.abs() < 1e-15 * root.abs() {
            break;
          }
        }
        return root;
      }
    }

    prev_val = next_val;
    x = next_x;

    // Safety: if we've searched very far, give up
    if x > n + (k as f64) * std::f64::consts::PI + 100.0 {
      break;
    }
  }

  f64::NAN
}

/// Compute Bessel J_n(z) using series expansion
pub fn bessel_j(n: f64, z: f64) -> f64 {
  // Handle negative integer orders: J_{-n}(z) = (-1)^n * J_n(z)
  if n < 0.0 && n == n.floor() {
    let n_abs = -n;
    let sign = if n_abs as i64 % 2 == 0 { 1.0 } else { -1.0 };
    return sign * bessel_j(n_abs, z);
  }

  // Special case: z = 0
  if z == 0.0 {
    return if n == 0.0 { 1.0 } else { 0.0 };
  }

  // Series: J_n(z) = sum_{m=0}^{inf} (-1)^m / (m! * Gamma(n+m+1)) * (z/2)^(2m+n)
  let half_z = z / 2.0;
  let mut sum = 0.0;
  let mut term = half_z.powf(n) / gamma_fn(n + 1.0);
  sum += term;

  for m in 1..300 {
    term *= -half_z * half_z / (m as f64 * (n + m as f64));
    sum += term;
    if term.abs() < 1e-17 * sum.abs().max(1e-300) {
      break;
    }
  }
  sum
}

/// BesselI[n, z] - Modified Bessel function of the first kind
pub fn bessel_i_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "BesselI expects exactly 2 arguments".into(),
    ));
  }
  let n_expr = &args[0];
  let z_expr = &args[1];

  // BesselI[n, 0] for integer n: I_0(0) = 1, I_n(0) = 0 for n != 0
  if matches!(z_expr, Expr::Integer(0))
    && let Expr::Integer(n) = n_expr
  {
    return if *n == 0 {
      Ok(Expr::Integer(1))
    } else {
      Ok(Expr::Integer(0))
    };
  }

  // Numeric evaluation
  let n_val = expr_to_f64(n_expr);
  let z_val = expr_to_f64(z_expr);
  let is_numeric_eval = n_val.is_some()
    && z_val.is_some()
    && (matches!(z_expr, Expr::Real(_)) || matches!(n_expr, Expr::Real(_)));

  if is_numeric_eval {
    let result = bessel_i(n_val.unwrap(), z_val.unwrap());
    return Ok(Expr::Real(result));
  }

  Ok(Expr::FunctionCall {
    name: "BesselI".to_string(),
    args: args.to_vec(),
  })
}

/// Compute I_n(z) using series: I_n(z) = Σ (z/2)^{2m+n} / (m! * Γ(n+m+1))
pub fn bessel_i(n: f64, z: f64) -> f64 {
  if n < 0.0 && n == n.floor() {
    // I_{-n}(z) = I_n(z) for integer n
    return bessel_i(-n, z);
  }

  if z == 0.0 {
    return if n == 0.0 { 1.0 } else { 0.0 };
  }

  let half_z = z / 2.0;
  let mut sum = 0.0;
  let mut term = half_z.powf(n) / gamma_fn(n + 1.0);
  sum += term;

  for m in 1..300 {
    term *= half_z * half_z / (m as f64 * (n + m as f64));
    sum += term;
    if term.abs() < 1e-17 * sum.abs().max(1e-300) {
      break;
    }
  }
  sum
}

/// BesselK[n, z] - Modified Bessel function of the second kind
pub fn bessel_k_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "BesselK expects exactly 2 arguments".into(),
    ));
  }
  let n_expr = &args[0];
  let z_expr = &args[1];

  // BesselK[n, 0]: K_0(0) = Infinity, K_n(0) = ComplexInfinity for n > 0
  if matches!(z_expr, Expr::Integer(0))
    && let Expr::Integer(n) = n_expr
  {
    return if *n == 0 {
      Ok(Expr::Identifier("Infinity".to_string()))
    } else {
      Ok(Expr::Identifier("ComplexInfinity".to_string()))
    };
  }

  // Numeric evaluation
  let n_val = expr_to_f64(n_expr);
  let z_val = expr_to_f64(z_expr);
  let is_numeric_eval = n_val.is_some()
    && z_val.is_some()
    && (matches!(z_expr, Expr::Real(_)) || matches!(n_expr, Expr::Real(_)));

  if is_numeric_eval {
    let result = bessel_k(n_val.unwrap(), z_val.unwrap());
    return Ok(Expr::Real(result));
  }

  Ok(Expr::FunctionCall {
    name: "BesselK".to_string(),
    args: args.to_vec(),
  })
}

/// Compute K_n(z) for integer n using the Temme method for small z
/// and the recurrence relation for integer orders.
pub fn bessel_k(n: f64, z: f64) -> f64 {
  let n_int = n.round() as i64;
  let n_abs = n_int.unsigned_abs();

  // K_{-n}(z) = K_n(z)
  let n_abs_f = n_abs as f64;

  // Compute K_0 and K_1 directly, then use recurrence for higher orders
  let (k0, k1) = bessel_k01(z);
  if n_abs == 0 {
    return k0;
  }
  if n_abs == 1 {
    return k1;
  }

  // Recurrence: K_{n+1}(z) = (2n/z) * K_n(z) + K_{n-1}(z)
  let mut km1 = k0;
  let mut k = k1;
  for i in 1..n_abs {
    let kp1 = (2.0 * i as f64 / z) * k + km1;
    km1 = k;
    k = kp1;
  }

  // If original n was non-integer, we'd need a different approach
  // but for now we handle integer n via recurrence
  if (n - n_abs_f).abs() > 1e-10 && n >= 0.0 {
    // Non-integer order: use K_v(z) = (π/2) * (I_{-v}(z) - I_v(z)) / sin(vπ)
    let pi = std::f64::consts::PI;
    let i_neg = bessel_i(-n, z);
    let i_pos = bessel_i(n, z);
    return (pi / 2.0) * (i_neg - i_pos) / (n * pi).sin();
  }

  k
}

/// Compute K_0(z) and K_1(z) simultaneously using the Temme series for small z
/// and asymptotic expansion for large z.
pub fn bessel_k01(z: f64) -> (f64, f64) {
  if z <= 2.0 {
    // Series for K_0 and K_1
    let euler_gamma = 0.5772156649015329;
    let lnz2 = (z / 2.0).ln();
    let t = z * z / 4.0;

    // K_0(z) = -(ln(z/2) + γ) * I_0(z) + Σ_{m=1}^∞ (z/2)^{2m} * H_m / (m!)^2
    let i0 = bessel_i(0.0, z);
    let mut k0 = -(lnz2 + euler_gamma) * i0;
    let mut term0 = 1.0;
    let mut hm = 0.0;
    for m in 1..100 {
      let mf = m as f64;
      term0 *= t / (mf * mf);
      hm += 1.0 / mf;
      k0 += term0 * hm;
      if (term0 * hm).abs() < 1e-17 * k0.abs().max(1e-300) {
        break;
      }
    }

    // K_1(z) = (1/z) + (ln(z/2) + γ - 1) * I_1(z)
    //   - (z/2) * Σ_{m=0}^∞ (z²/4)^m * (H_m + H_{m+1}) / (2 * m! * (m+1)!)
    // But the correct form is:
    // K_1(z) = (1/z) - (ln(z/2) + γ) * I_1(z) + (z/4) * Σ_{m=0}^∞ ...
    // Let's use recurrence from K_0 instead: K_1 = -K_0' (for the derivative)
    // Or more reliably, compute via the Wronskian: I_n * K_{n+1} + I_{n+1} * K_n = 1/z
    // So K_1 = (1/z - I_1 * K_0) / I_0
    let i1 = bessel_i(1.0, z);
    let k1 = (1.0 / z - i1 * k0) / i0;

    (k0, k1)
  } else {
    // Asymptotic: K_n(z) ~ sqrt(π/(2z)) * e^{-z} * P_n(z)
    // where P_n(z) = 1 + (4n²-1)/(8z) + (4n²-1)(4n²-9)/(2!(8z)²) + ...
    let pi = std::f64::consts::PI;
    let prefactor = (pi / (2.0 * z)).sqrt() * (-z).exp();

    // K_0 asymptotic (μ = 0)
    let mut sum0 = 1.0;
    let mut term0 = 1.0;
    for m in 1..30 {
      let mf = m as f64;
      let factor = -(0.0 - (2.0 * mf - 1.0).powi(2)) / (8.0 * z * mf);
      term0 *= factor;
      if term0.abs() < 1e-17 {
        break;
      }
      sum0 += term0;
    }

    // K_1 asymptotic (μ = 4)
    let mut sum1 = 1.0;
    let mut term1 = 1.0;
    for m in 1..30 {
      let mf = m as f64;
      let factor = (4.0 - (2.0 * mf - 1.0).powi(2)) / (8.0 * z * mf);
      term1 *= factor;
      if term1.abs() < 1e-17 {
        break;
      }
      sum1 += term1;
    }

    (prefactor * sum0, prefactor * sum1)
  }
}

/// Compute Jacobi elliptic functions sn(u, m), cn(u, m), dn(u, m) numerically
/// using the descending Landen (AGM) transformation.
/// Parameter m is the parameter (not the modulus k; m = k^2).
pub fn jacobi_elliptic(u: f64, m: f64) -> (f64, f64, f64) {
  // Edge cases
  if m.abs() < 1e-16 {
    // m = 0: sn = sin(u), cn = cos(u), dn = 1
    return (u.sin(), u.cos(), 1.0);
  }
  if (m - 1.0).abs() < 1e-16 {
    // m = 1: sn = tanh(u), cn = sech(u), dn = sech(u)
    let s = u.tanh();
    let c = 1.0 / u.cosh();
    return (s, c, c);
  }

  // AGM iteration to compute the sequence of a_n, b_n, c_n
  let mut a = vec![1.0];
  let mut b = vec![(1.0 - m).sqrt()];
  let mut c = vec![m.sqrt()];

  let n_max = 50;
  for _ in 0..n_max {
    let a_prev = *a.last().unwrap();
    let b_prev = *b.last().unwrap();
    a.push((a_prev + b_prev) / 2.0);
    b.push((a_prev * b_prev).sqrt());
    c.push((a_prev - b_prev) / 2.0);

    if c.last().unwrap().abs() < 1e-16 {
      break;
    }
  }

  let n = a.len() - 1;
  // phi_N = 2^N * a_N * u
  let mut phi = (1u64 << n as u64) as f64 * a[n] * u;

  // Descend: phi_{n-1} from phi_n
  for i in (1..=n).rev() {
    phi = (phi + (c[i] / a[i] * phi.sin()).asin()) / 2.0;
  }

  let sn = phi.sin();
  let cn = phi.cos();
  let dn = (1.0 - m * sn * sn).sqrt();

  (sn, cn, dn)
}

/// Hypergeometric0F1[a, z] - confluent hypergeometric limit function
/// 0F1(a; z) = Σ z^k / (k! * Pochhammer(a,k)) for k = 0, 1, 2, ...
pub fn hypergeometric_0f1_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Hypergeometric0F1 expects exactly 2 arguments".into(),
    ));
  }

  let a_expr = &args[0];
  let z_expr = &args[1];

  // Hypergeometric0F1[a, 0] = 1
  if is_expr_zero(z_expr) {
    return Ok(Expr::Integer(1));
  }

  // Numeric evaluation
  let a_val = try_eval_to_f64(a_expr);
  let z_val = try_eval_to_f64(z_expr);

  if let (Some(a), Some(z)) = (a_val, z_val)
    && (matches!(a_expr, Expr::Real(_)) || matches!(z_expr, Expr::Real(_)))
  {
    return Ok(Expr::Real(hypergeometric_0f1_f64(a, z)));
  }

  Ok(Expr::FunctionCall {
    name: "Hypergeometric0F1".to_string(),
    args: args.to_vec(),
  })
}

/// Compute 0F1(a; z) numerically via series expansion
pub fn hypergeometric_0f1_f64(a: f64, z: f64) -> f64 {
  let mut sum = 1.0;
  let mut term = 1.0;
  for k in 0..200 {
    let kf = k as f64;
    term *= z / ((kf + 1.0) * (a + kf));
    sum += term;
    if term.abs() < 1e-16 * sum.abs() {
      break;
    }
  }
  sum
}

/// Hypergeometric0F1Regularized[a, z] — regularized confluent hypergeometric limit function.
///
/// 0F1~(a; z) = sum_{k=0}^{inf} z^k / (Gamma(a + k) * k!)
pub fn hypergeometric_0f1_regularized_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Hypergeometric0F1Regularized expects exactly 2 arguments".into(),
    ));
  }

  let a_expr = &args[0];
  let z_expr = &args[1];

  // Hypergeometric0F1Regularized[a, 0] = 1/Gamma(a)
  // For non-negative integer a=0: 1/Gamma(0) = 0
  // For positive integer a: 1/Gamma(a) is well-defined
  if is_expr_zero(z_expr)
    && let Some(a_val) = try_eval_to_f64(a_expr)
  {
    let ga = gamma_fn(a_val);
    if ga.is_infinite() || ga == 0.0 {
      return Ok(Expr::Integer(0));
    }
    if a_val == a_val.floor() && a_val > 0.0 {
      // 1/Gamma(a) for positive integer a — return exact integer if possible
      return Ok(Expr::Integer(1));
    }
  }

  // Numeric evaluation
  let a_val = try_eval_to_f64(a_expr);
  let z_val = try_eval_to_f64(z_expr);

  if let (Some(a), Some(z)) = (a_val, z_val)
    && (matches!(a_expr, Expr::Real(_)) || matches!(z_expr, Expr::Real(_)))
  {
    return Ok(Expr::Real(hypergeometric_0f1_regularized_f64(a, z)));
  }

  Ok(Expr::FunctionCall {
    name: "Hypergeometric0F1Regularized".to_string(),
    args: args.to_vec(),
  })
}

/// Compute regularized 0F1~(a; z) = sum_{k=0}^{inf} z^k / (Gamma(a + k) * k!)
pub fn hypergeometric_0f1_regularized_f64(a: f64, z: f64) -> f64 {
  let mut sum = 0.0;
  let mut z_power = 1.0; // z^k
  let mut factorial = 1.0; // k!

  for k in 0..300 {
    if k > 0 {
      z_power *= z;
      factorial *= k as f64;
    }
    let ga_k = gamma_fn(a + k as f64);
    if ga_k.is_infinite() || ga_k.is_nan() {
      // Skip terms where Gamma(a+k) diverges (non-positive integer a+k)
      continue;
    }
    let term = z_power / (ga_k * factorial);
    sum += term;
    if k > 0 && term.abs() < 1e-16 * sum.abs().max(1e-300) {
      break;
    }
  }
  sum
}

/// JacobiAmplitude[u, m] - amplitude for Jacobi elliptic functions
/// am(u, m) = arcsin(sn(u, m)), inverse of EllipticF
pub fn jacobi_amplitude_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiAmplitude expects exactly 2 arguments".into(),
    ));
  }

  let u = &args[0];
  let m = &args[1];

  // JacobiAmplitude[0, m] = 0
  if is_expr_zero(u) {
    if has_real_arg(u, m) {
      return Ok(Expr::Real(0.0));
    }
    return Ok(Expr::Integer(0));
  }

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (try_eval_to_f64(u), try_eval_to_f64(m))
    && (matches!(u, Expr::Real(_)) || matches!(m, Expr::Real(_)))
  {
    let (sn, cn, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(sn.atan2(cn)));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiAmplitude".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiDN[u, m] - Jacobi elliptic function dn
pub fn jacobi_dn_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiDN expects exactly 2 arguments".into(),
    ));
  }

  let u = &args[0];
  let m = &args[1];

  // JacobiDN[0, m] = 1
  if is_expr_zero(u) {
    return Ok(Expr::Integer(1));
  }

  // JacobiDN[u, 0] = 1
  if is_expr_zero(m) {
    return Ok(Expr::Integer(1));
  }

  // JacobiDN[u, 1] = Sech[u]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Sech".to_string(),
      args: vec![u.clone()],
    });
  }

  // JacobiDN[-u, m] = JacobiDN[u, m] (even function)
  if let Some(inner) = extract_negated_expr(u) {
    return Ok(Expr::FunctionCall {
      name: "JacobiDN".to_string(),
      args: vec![inner, m.clone()],
    });
  }

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (_, _, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(dn));
  }

  // Unevaluated
  Ok(Expr::FunctionCall {
    name: "JacobiDN".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiSN[u, m] - Jacobi elliptic function sn
pub fn jacobi_sn_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiSN expects exactly 2 arguments".into(),
    ));
  }

  let u = &args[0];
  let m = &args[1];

  // JacobiSN[0, m] = 0
  if is_expr_zero(u) {
    return Ok(Expr::Integer(0));
  }

  // JacobiSN[u, 0] = Sin[u]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Sin".to_string(),
      args: vec![u.clone()],
    });
  }

  // JacobiSN[u, 1] = Tanh[u]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Tanh".to_string(),
      args: vec![u.clone()],
    });
  }

  // JacobiSN[-u, m] = -JacobiSN[u, m] (odd function)
  if let Some(inner) = extract_negated_expr(u) {
    return Ok(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(-1)),
      right: Box::new(Expr::FunctionCall {
        name: "JacobiSN".to_string(),
        args: vec![inner, m.clone()],
      }),
    });
  }

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (sn, _, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(sn));
  }

  // Unevaluated
  Ok(Expr::FunctionCall {
    name: "JacobiSN".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiCN[u, m] - Jacobi elliptic function cn
pub fn jacobi_cn_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiCN expects exactly 2 arguments".into(),
    ));
  }

  let u = &args[0];
  let m = &args[1];

  // JacobiCN[0, m] = 1
  if is_expr_zero(u) {
    return Ok(Expr::Integer(1));
  }

  // JacobiCN[u, 0] = Cos[u]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Cos".to_string(),
      args: vec![u.clone()],
    });
  }

  // JacobiCN[u, 1] = Sech[u]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Sech".to_string(),
      args: vec![u.clone()],
    });
  }

  // JacobiCN[-u, m] = JacobiCN[u, m] (even function)
  if let Some(inner) = extract_negated_expr(u) {
    return Ok(Expr::FunctionCall {
      name: "JacobiCN".to_string(),
      args: vec![inner, m.clone()],
    });
  }

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (_, cn, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(cn));
  }

  // Unevaluated
  Ok(Expr::FunctionCall {
    name: "JacobiCN".to_string(),
    args: args.to_vec(),
  })
}

// ─── InverseJacobi functions ────────────────────────────────────────

/// Check if either argument is a Real (for numeric dispatch)
fn has_real_arg(a: &Expr, b: &Expr) -> bool {
  matches!(a, Expr::Real(_)) || matches!(b, Expr::Real(_))
}

/// Helper: compute InverseJacobiSN numerically via EllipticF[ArcSin[v], m]
fn inverse_jacobi_sn_numeric(v: f64, m: f64) -> f64 {
  elliptic_f(v.asin(), m)
}

/// InverseJacobiSN[v, m]
pub fn inverse_jacobi_sn_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiSN expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // InverseJacobiSN[0, m] = 0
  if is_expr_zero(v) {
    if has_real_arg(v, m) {
      return Ok(Expr::Real(0.0));
    }
    return Ok(Expr::Integer(0));
  }

  // InverseJacobiSN[1, m] = EllipticK[m]
  if is_expr_one(v) && !has_real_arg(v, m) {
    return Ok(Expr::FunctionCall {
      name: "EllipticK".to_string(),
      args: vec![m.clone()],
    });
  }

  // Numeric evaluation: EllipticF[ArcSin[v], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && (has_real_arg(v, m) || is_expr_one(v))
  {
    return Ok(Expr::Real(inverse_jacobi_sn_numeric(v_f, m_f)));
  }

  // InverseJacobiSN[x, 0] = ArcSin[x]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcSin".to_string(),
      args: vec![v.clone()],
    });
  }

  // InverseJacobiSN[x, 1] = ArcTanh[x]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcTanh".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiSN".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiCN[v, m]
pub fn inverse_jacobi_cn_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiCN expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // InverseJacobiCN[1, m] = 0
  if is_expr_one(v) {
    return Ok(Expr::Integer(0));
  }

  // InverseJacobiCN[0, m] = EllipticK[m] (symbolic only)
  if is_expr_zero(v) && !has_real_arg(v, m) {
    return Ok(Expr::FunctionCall {
      name: "EllipticK".to_string(),
      args: vec![m.clone()],
    });
  }

  // Numeric: EllipticF[ArcCos[v], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
  {
    return Ok(Expr::Real(elliptic_f(v_f.acos(), m_f)));
  }

  // InverseJacobiCN[x, 0] = ArcCos[x]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcCos".to_string(),
      args: vec![v.clone()],
    });
  }

  // InverseJacobiCN[x, 1] = ArcSech[x]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcSech".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiCN".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiDN[v, m]
pub fn inverse_jacobi_dn_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiDN expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // InverseJacobiDN[1, m] = 0
  if is_expr_one(v) {
    return Ok(Expr::Integer(0));
  }

  // Numeric: EllipticF[ArcSin[Sqrt[(1-v^2)/m]], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
    && m_f != 0.0
  {
    let arg = ((1.0 - v_f * v_f) / m_f).sqrt();
    return Ok(Expr::Real(elliptic_f(arg.asin(), m_f)));
  }

  // InverseJacobiDN[x, 1] = ArcSech[x]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcSech".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiDN".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiCD[v, m]
pub fn inverse_jacobi_cd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiCD expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // InverseJacobiCD[1, m] = 0
  if is_expr_one(v) {
    return Ok(Expr::Integer(0));
  }

  // InverseJacobiCD[0, m] = EllipticK[m] (symbolic only)
  if is_expr_zero(v) && !has_real_arg(v, m) {
    return Ok(Expr::FunctionCall {
      name: "EllipticK".to_string(),
      args: vec![m.clone()],
    });
  }

  // Numeric: EllipticF[ArcSin[Sqrt[(1-v^2)/(1-m*v^2)]], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
  {
    let arg = ((1.0 - v_f * v_f) / (1.0 - m_f * v_f * v_f)).sqrt();
    return Ok(Expr::Real(elliptic_f(arg.asin(), m_f)));
  }

  // InverseJacobiCD[x, 0] = ArcCos[x]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcCos".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiCD".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiSC[v, m]
pub fn inverse_jacobi_sc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiSC expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // InverseJacobiSC[0, m] = 0
  if is_expr_zero(v) {
    if has_real_arg(v, m) {
      return Ok(Expr::Real(0.0));
    }
    return Ok(Expr::Integer(0));
  }

  // Numeric: EllipticF[ArcSin[v/Sqrt[1+v^2]], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
  {
    let arg = v_f / (1.0 + v_f * v_f).sqrt();
    return Ok(Expr::Real(elliptic_f(arg.asin(), m_f)));
  }

  // InverseJacobiSC[x, 0] = ArcTan[x]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcTan".to_string(),
      args: vec![v.clone()],
    });
  }

  // InverseJacobiSC[x, 1] = ArcSinh[x]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcSinh".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiSC".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiCS[v, m]
pub fn inverse_jacobi_cs_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiCS expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // Numeric: EllipticF[ArcSin[1/Sqrt[1+v^2]], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
  {
    let arg = 1.0 / (1.0 + v_f * v_f).sqrt();
    return Ok(Expr::Real(elliptic_f(arg.asin(), m_f)));
  }

  // InverseJacobiCS[x, 0] = ArcCot[x]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcCot".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiCS".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiSD[v, m]
pub fn inverse_jacobi_sd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiSD expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // InverseJacobiSD[0, m] = 0
  if is_expr_zero(v) {
    if has_real_arg(v, m) {
      return Ok(Expr::Real(0.0));
    }
    return Ok(Expr::Integer(0));
  }

  // Numeric: EllipticF[ArcSin[v/Sqrt[1+m*v^2]], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
  {
    let arg = v_f / (1.0 + m_f * v_f * v_f).sqrt();
    return Ok(Expr::Real(elliptic_f(arg.asin(), m_f)));
  }

  // InverseJacobiSD[x, 0] = ArcSin[x]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcSin".to_string(),
      args: vec![v.clone()],
    });
  }

  // InverseJacobiSD[x, 1] = ArcSinh[x]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcSinh".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiSD".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiDS[v, m]
pub fn inverse_jacobi_ds_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiDS expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // Numeric: EllipticF[ArcSin[1/Sqrt[v^2+m]], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
  {
    let arg = 1.0 / (v_f * v_f + m_f).sqrt();
    return Ok(Expr::Real(elliptic_f(arg.asin(), m_f)));
  }

  // InverseJacobiDS[x, 0] = ArcCsc[x]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcCsc".to_string(),
      args: vec![v.clone()],
    });
  }

  // InverseJacobiDS[x, 1] = ArcCsch[x]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcCsch".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiDS".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiNS[v, m]
pub fn inverse_jacobi_ns_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiNS expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // Numeric: EllipticF[ArcSin[1/v], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
  {
    let arg = 1.0 / v_f;
    return Ok(Expr::Real(elliptic_f(arg.asin(), m_f)));
  }

  // InverseJacobiNS[x, 0] = ArcCsc[x]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcCsc".to_string(),
      args: vec![v.clone()],
    });
  }

  // InverseJacobiNS[x, 1] = ArcCoth[x]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcCoth".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiNS".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiNC[v, m]
pub fn inverse_jacobi_nc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiNC expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // InverseJacobiNC[1, m] = 0
  if is_expr_one(v) {
    return Ok(Expr::Integer(0));
  }

  // Numeric: EllipticF[ArcSin[Sqrt[1-1/v^2]], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
  {
    let arg = (1.0 - 1.0 / (v_f * v_f)).sqrt();
    return Ok(Expr::Real(elliptic_f(arg.asin(), m_f)));
  }

  // InverseJacobiNC[x, 0] = ArcSec[x]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcSec".to_string(),
      args: vec![v.clone()],
    });
  }

  // InverseJacobiNC[x, 1] = ArcCosh[x]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcCosh".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiNC".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiND[v, m]
pub fn inverse_jacobi_nd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiND expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // InverseJacobiND[1, m] = 0
  if is_expr_one(v) {
    return Ok(Expr::Integer(0));
  }

  // Numeric: EllipticF[ArcSin[Sqrt[(v^2-1)/(m*v^2)]], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
    && m_f != 0.0
  {
    let arg = ((v_f * v_f - 1.0) / (m_f * v_f * v_f)).sqrt();
    return Ok(Expr::Real(elliptic_f(arg.asin(), m_f)));
  }

  // InverseJacobiND[x, 1] = ArcCosh[x]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcCosh".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiND".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiDC[v, m]
pub fn inverse_jacobi_dc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiDC expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // InverseJacobiDC[1, m] = 0
  if is_expr_one(v) {
    return Ok(Expr::Integer(0));
  }

  // Numeric: EllipticF[ArcSin[Sqrt[(v^2-1)/(v^2-m)]], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
  {
    let arg = ((v_f * v_f - 1.0) / (v_f * v_f - m_f)).sqrt();
    return Ok(Expr::Real(elliptic_f(arg.asin(), m_f)));
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiDC".to_string(),
    args: args.to_vec(),
  })
}

/// Extract the inner expression from a negated expression like Times[-1, x] or -x
pub fn extract_negated_expr(expr: &Expr) -> Option<Expr> {
  match expr {
    // Times[-1, x] form (FunctionCall)
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() == 2
        && matches!(args[0], Expr::Integer(-1)) =>
    {
      Some(args[1].clone())
    }
    // BinaryOp Times[-1, x] form
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if matches!(left.as_ref(), Expr::Integer(-1)) {
        Some(right.as_ref().clone())
      } else if matches!(right.as_ref(), Expr::Integer(-1)) {
        Some(left.as_ref().clone())
      } else {
        None
      }
    }
    // UnaryOp Minus form
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Some(operand.as_ref().clone()),
    _ => None,
  }
}

/// JacobiSC[u, m] - Jacobi SC elliptic function (SN/CN)
pub fn jacobi_sc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiSC expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  // JacobiSC[0, m] = 0
  if is_expr_zero(u) {
    if has_real_arg(u, m) {
      return Ok(Expr::Real(0.0));
    }
    return Ok(Expr::Integer(0));
  }
  // JacobiSC[u, 0] = Tan[u]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Tan".to_string(),
      args: vec![u.clone()],
    });
  }
  // JacobiSC[u, 1] = Sinh[u]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Sinh".to_string(),
      args: vec![u.clone()],
    });
  }

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (sn, cn, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(sn / cn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiSC".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiDC[u, m] - Jacobi DC elliptic function (DN/CN)
pub fn jacobi_dc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiDC expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  // JacobiDC[0, m] = 1
  if is_expr_zero(u) {
    if has_real_arg(u, m) {
      return Ok(Expr::Real(1.0));
    }
    return Ok(Expr::Integer(1));
  }
  // JacobiDC[u, 0] = Sec[u]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Sec".to_string(),
      args: vec![u.clone()],
    });
  }
  // JacobiDC[u, 1] = 1
  if is_expr_one(m) {
    return Ok(Expr::Integer(1));
  }

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (_, cn, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(dn / cn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiDC".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiCD[u, m] - Jacobi CD elliptic function (CN/DN)
pub fn jacobi_cd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiCD expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  // JacobiCD[0, m] = 1
  if is_expr_zero(u) {
    if has_real_arg(u, m) {
      return Ok(Expr::Real(1.0));
    }
    return Ok(Expr::Integer(1));
  }
  // JacobiCD[u, 0] = Cos[u]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Cos".to_string(),
      args: vec![u.clone()],
    });
  }
  // JacobiCD[u, 1] = 1
  if is_expr_one(m) {
    return Ok(Expr::Integer(1));
  }

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (_, cn, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(cn / dn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiCD".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiSD[u, m] - Jacobi SD elliptic function (SN/DN)
pub fn jacobi_sd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiSD expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  // JacobiSD[0, m] = 0
  if is_expr_zero(u) {
    if has_real_arg(u, m) {
      return Ok(Expr::Real(0.0));
    }
    return Ok(Expr::Integer(0));
  }
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Sin".to_string(),
      args: vec![u.clone()],
    });
  }
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Sinh".to_string(),
      args: vec![u.clone()],
    });
  }

  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (sn, _, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(sn / dn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiSD".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiCS[u, m] - Jacobi CS elliptic function (CN/SN)
pub fn jacobi_cs_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiCS expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  if is_expr_zero(u) {
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Cot".to_string(),
      args: vec![u.clone()],
    });
  }
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Csch".to_string(),
      args: vec![u.clone()],
    });
  }

  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (sn, cn, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(cn / sn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiCS".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiDS[u, m] - Jacobi DS elliptic function (DN/SN)
pub fn jacobi_ds_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiDS expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  if is_expr_zero(u) {
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Csc".to_string(),
      args: vec![u.clone()],
    });
  }
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Csch".to_string(),
      args: vec![u.clone()],
    });
  }

  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (sn, _, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(dn / sn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiDS".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiNS[u, m] - Jacobi NS elliptic function (1/SN)
pub fn jacobi_ns_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiNS expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  if is_expr_zero(u) {
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Csc".to_string(),
      args: vec![u.clone()],
    });
  }
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Coth".to_string(),
      args: vec![u.clone()],
    });
  }

  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (sn, _, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(1.0 / sn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiNS".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiND[u, m] - Jacobi ND elliptic function (1/DN)
pub fn jacobi_nd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiND expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  // JacobiND[0, m] = 1
  if is_expr_zero(u) {
    if has_real_arg(u, m) {
      return Ok(Expr::Real(1.0));
    }
    return Ok(Expr::Integer(1));
  }
  if is_expr_zero(m) {
    return Ok(Expr::Integer(1));
  }
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Cosh".to_string(),
      args: vec![u.clone()],
    });
  }

  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (_, _, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(1.0 / dn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiND".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiNC[u, m] - Jacobi NC elliptic function (1/CN)
pub fn jacobi_nc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiNC expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  // JacobiNC[0, m] = 1
  if is_expr_zero(u) {
    if has_real_arg(u, m) {
      return Ok(Expr::Real(1.0));
    }
    return Ok(Expr::Integer(1));
  }
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Sec".to_string(),
      args: vec![u.clone()],
    });
  }
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Cosh".to_string(),
      args: vec![u.clone()],
    });
  }

  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (_, cn, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(1.0 / cn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiNC".to_string(),
    args: args.to_vec(),
  })
}

/// Check if an expression is numerically zero
pub fn is_expr_zero(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(0) => true,
    Expr::Real(f) => *f == 0.0,
    _ => false,
  }
}

/// Check if an expression is numerically one
pub fn is_expr_one(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(1) => true,
    Expr::Real(f) => *f == 1.0,
    _ => false,
  }
}

/// Check if an expression represents -Infinity
pub fn is_neg_infinity(expr: &Expr) -> bool {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Times"
    && args.len() == 2
    && let (Expr::Integer(-1), Expr::Identifier(s)) = (&args[0], &args[1])
  {
    return s == "Infinity";
  }
  if let Expr::BinaryOp {
    op: BinaryOperator::Times,
    left,
    right,
  } = expr
    && let (Expr::Integer(-1), Expr::Identifier(s)) =
      (left.as_ref(), right.as_ref())
  {
    return s == "Infinity";
  }
  if let Expr::UnaryOp {
    op: UnaryOperator::Minus,
    operand,
  } = expr
    && let Expr::Identifier(s) = operand.as_ref()
  {
    return s == "Infinity";
  }
  false
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

/// BesselY[n, z] - Bessel function of the second kind
pub fn bessel_y_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "BesselY expects exactly 2 arguments".into(),
    ));
  }
  let n_expr = &args[0];
  let z_expr = &args[1];

  // BesselY[n, 0]: Y_0(0) = -Infinity, Y_n(0) = ComplexInfinity for n > 0
  if matches!(z_expr, Expr::Integer(0))
    && let Expr::Integer(n) = n_expr
  {
    return if *n == 0 {
      Ok(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), Expr::Identifier("Infinity".to_string())],
      })
    } else {
      Ok(Expr::Identifier("ComplexInfinity".to_string()))
    };
  }

  // Numeric evaluation
  let n_val = expr_to_f64(n_expr);
  let z_val = expr_to_f64(z_expr);
  let is_numeric_eval = n_val.is_some()
    && z_val.is_some()
    && (matches!(z_expr, Expr::Real(_)) || matches!(n_expr, Expr::Real(_)));

  if is_numeric_eval {
    let result = bessel_y(n_val.unwrap(), z_val.unwrap());
    return Ok(Expr::Real(result));
  }

  Ok(Expr::FunctionCall {
    name: "BesselY".to_string(),
    args: args.to_vec(),
  })
}

/// Compute Bessel Y_n(z) (second kind)
pub fn bessel_y(n: f64, z: f64) -> f64 {
  // Handle negative integer orders: Y_{-n}(z) = (-1)^n * Y_n(z)
  if n < 0.0 && n == n.floor() {
    let n_abs = -n;
    let sign = if n_abs as i64 % 2 == 0 { 1.0 } else { -1.0 };
    return sign * bessel_y(n_abs, z);
  }

  if n == n.floor() && n >= 0.0 {
    // Integer order: use Y_0, Y_1 from series, then recurrence
    let n_int = n as i64;
    let y0 = bessel_y0(z);
    if n_int == 0 {
      return y0;
    }
    let y1 = bessel_y1(z);
    if n_int == 1 {
      return y1;
    }
    // Recurrence: Y_{n+1}(z) = (2n/z) * Y_n(z) - Y_{n-1}(z)
    let mut y_prev = y0;
    let mut y_curr = y1;
    for k in 1..n_int {
      let y_next = (2.0 * k as f64 / z) * y_curr - y_prev;
      y_prev = y_curr;
      y_curr = y_next;
    }
    y_curr
  } else {
    // Non-integer order: Y_n(z) = (J_n(z)*cos(nπ) - J_{-n}(z)) / sin(nπ)
    let j_n = bessel_j(n, z);
    let j_neg_n = bessel_j(-n, z);
    let n_pi = n * std::f64::consts::PI;
    (j_n * n_pi.cos() - j_neg_n) / n_pi.sin()
  }
}

/// Y_0(z) = (2/π) * (J_0(z) * (ln(z/2) + γ) + Σ (-1)^{m+1} H_m * (z/2)^{2m} / (m!)^2)
pub fn bessel_y0(z: f64) -> f64 {
  let two_over_pi = 2.0 / std::f64::consts::PI;
  let euler_gamma = 0.5772156649015329;
  let half_z = z / 2.0;

  let j0 = bessel_j(0.0, z);

  // Series part: Σ_{m=1}^{∞} (-1)^{m+1} * H_m * (z/2)^{2m} / (m!)^2
  let mut series_sum = 0.0;
  let mut term = 1.0; // (z/2)^{2m} / (m!)^2 starting value
  let mut h_m = 0.0; // harmonic number H_m

  for m in 1..300 {
    let mf = m as f64;
    h_m += 1.0 / mf;
    term *= half_z * half_z / (mf * mf);
    let sign = if m % 2 == 0 { -1.0 } else { 1.0 };
    series_sum += sign * h_m * term;
    if (h_m * term).abs() < 1e-17 * series_sum.abs().max(1e-300) {
      break;
    }
  }

  two_over_pi * (j0 * (half_z.ln() + euler_gamma) + series_sum)
}

/// Y_1(z) = (2/π) * (J_1(z) * ln(z/2) - 1/z + Σ ...)
pub fn bessel_y1(z: f64) -> f64 {
  let two_over_pi = 2.0 / std::f64::consts::PI;
  let euler_gamma = 0.5772156649015329;
  let half_z = z / 2.0;

  let j1 = bessel_j(1.0, z);

  // Y_1(z) = (2/π) * (J_1(z)*(ln(z/2) + γ) - 1/z - (1/2) Σ_{m=0}^∞ (-1)^m (H_m + H_{m+1}) (z/2)^{2m+1} / (m!(m+1)!))
  let mut series_sum = 0.0;
  let mut term = half_z; // (z/2)^{2m+1} / (m! * (m+1)!) starting with m=0: (z/2)^1 / (0! * 1!) = z/2
  let mut h_m = 0.0; // H_0 = 0
  let mut h_m1 = 1.0; // H_1 = 1

  series_sum += (h_m + h_m1) * term; // m=0 term (sign = +1 for (-1)^0)

  for m in 1..300 {
    let mf = m as f64;
    h_m += 1.0 / mf;
    h_m1 += 1.0 / (mf + 1.0);
    term *= -half_z * half_z / (mf * (mf + 1.0));
    series_sum += (h_m + h_m1) * term;
    if ((h_m + h_m1) * term).abs() < 1e-17 * series_sum.abs().max(1e-300) {
      break;
    }
  }

  two_over_pi
    * (j1 * ((half_z).ln() + euler_gamma) - 1.0 / z - 0.5 * series_sum)
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

/// WeierstrassP[u, {g2, g3}] - Weierstrass elliptic function ℘(u; g₂, g₃)
pub fn weierstrass_p_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "WeierstrassP expects exactly 2 arguments".into(),
    ));
  }

  let u = &args[0];
  let (g2, g3) = match &args[1] {
    Expr::List(items) if items.len() == 2 => (&items[0], &items[1]),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "WeierstrassP".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Special case: u = 0 → ComplexInfinity (pole at origin)
  if is_expr_zero(u) {
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }

  // Numeric evaluation
  if let (Some(u_f), Some(g2_f), Some(g3_f)) =
    (try_eval_to_f64(u), try_eval_to_f64(g2), try_eval_to_f64(g3))
  {
    let result = weierstrass_p_numeric(u_f, g2_f, g3_f);
    return Ok(Expr::Real(result));
  }

  // Symbolic: return unevaluated
  Ok(Expr::FunctionCall {
    name: "WeierstrassP".to_string(),
    args: args.to_vec(),
  })
}

/// WeierstrassPPrime[u, {g2, g3}] - Derivative of Weierstrass elliptic function ℘'(u; g₂, g₃)
pub fn weierstrass_p_prime_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "WeierstrassPPrime expects exactly 2 arguments".into(),
    ));
  }

  let u = &args[0];
  let (g2, g3) = match &args[1] {
    Expr::List(items) if items.len() == 2 => (&items[0], &items[1]),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "WeierstrassPPrime".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Special case: u = 0 → ComplexInfinity (pole of order 3 at origin)
  if is_expr_zero(u) {
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }

  // Numeric evaluation
  if let (Some(u_f), Some(g2_f), Some(g3_f)) =
    (try_eval_to_f64(u), try_eval_to_f64(g2), try_eval_to_f64(g3))
  {
    let result = weierstrass_p_prime_numeric(u_f, g2_f, g3_f);
    return Ok(Expr::Real(result));
  }

  // Symbolic: return unevaluated
  Ok(Expr::FunctionCall {
    name: "WeierstrassPPrime".to_string(),
    args: args.to_vec(),
  })
}

/// Compute WeierstrassPPrime numerically via central difference of WeierstrassP
pub fn weierstrass_p_prime_numeric(u: f64, g2: f64, g3: f64) -> f64 {
  // Use central difference with step size chosen for good accuracy
  let h = u.abs().max(1.0) * 1e-6;
  let p_plus = weierstrass_p_numeric(u + h, g2, g3);
  let p_minus = weierstrass_p_numeric(u - h, g2, g3);
  (p_plus - p_minus) / (2.0 * h)
}

/// Compute WeierstrassP numerically using cubic roots + Jacobi elliptic functions
pub fn weierstrass_p_numeric(u: f64, g2: f64, g3: f64) -> f64 {
  // Solve depressed cubic: t³ - (g2/4)t - (g3/4) = 0
  let p = -g2 / 4.0;
  let q = -g3 / 4.0;

  // Cubic discriminant: Δ = -4p³ - 27q²
  let delta = -4.0 * p * p * p - 27.0 * q * q;

  if delta >= 0.0 && p < -1e-16 {
    // Three real roots — use trigonometric method
    let mp3 = -p / 3.0; // > 0
    let r = mp3.sqrt();
    let cos_arg = (-q / (2.0 * mp3 * r)).clamp(-1.0, 1.0);
    let alpha = cos_arg.acos();

    let pi = std::f64::consts::PI;
    let mut roots = [
      2.0 * r * (alpha / 3.0).cos(),
      2.0 * r * ((alpha + 2.0 * pi) / 3.0).cos(),
      2.0 * r * ((alpha + 4.0 * pi) / 3.0).cos(),
    ];
    roots.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let e1 = roots[0];
    let e2 = roots[1];
    let e3 = roots[2];

    let denom = e1 - e3;
    if denom.abs() < 1e-300 {
      return weierstrass_p_laurent(u, g2, g3);
    }

    let m = (e2 - e3) / denom;
    let z = denom.sqrt() * u;
    let (sn, _, _) = jacobi_elliptic(z, m);

    if sn.abs() < 1e-300 {
      return f64::INFINITY;
    }
    e3 + denom / (sn * sn)
  } else {
    // One real root or degenerate — use duplication formula with Laurent series
    // The Laurent series converges only for small |u|, so we halve u
    // repeatedly until it's small enough, then apply the duplication formula.
    weierstrass_p_duplication(u, g2, g3)
  }
}

/// Compute ℘(u) using repeated argument halving + duplication formula.
/// The duplication formula is: ℘(2z) = -2℘(z) + (6℘(z)² - g₂/2)² / (4(4℘(z)³ - g₂℘(z) - g₃))
fn weierstrass_p_duplication(u: f64, g2: f64, g3: f64) -> f64 {
  // Halve u until it's small enough for the Laurent series to converge well
  let threshold = 0.3;
  let mut halvings = 0u32;
  let mut z = u;
  while z.abs() > threshold && halvings < 60 {
    z /= 2.0;
    halvings += 1;
  }

  // Compute ℘(z) using Laurent series (converges well for small z)
  let mut wp = weierstrass_p_laurent(z, g2, g3);

  // Apply duplication formula to double back: ℘(2z) from ℘(z)
  for _ in 0..halvings {
    let wp2 = wp * wp;
    let wp3 = wp2 * wp;
    let denom = 4.0 * wp3 - g2 * wp - g3; // = ℘'(z)²
    if denom.abs() < 1e-300 {
      return f64::INFINITY;
    }
    let num = 6.0 * wp2 - g2 / 2.0; // = ℘''(z)
    wp = -2.0 * wp + num * num / (4.0 * denom);
  }

  wp
}

/// Laurent series fallback: ℘(u) = 1/u² + Σ cₖ u^{2k}
pub fn weierstrass_p_laurent(u: f64, g2: f64, g3: f64) -> f64 {
  let max_terms = 30;
  let mut c = vec![0.0; max_terms];
  if max_terms > 1 {
    c[1] = g2 / 20.0;
  }
  if max_terms > 2 {
    c[2] = g3 / 28.0;
  }
  for k in 3..max_terms {
    let mut sum = 0.0;
    for j in 1..=(k - 2) {
      sum += c[j] * c[k - 1 - j];
    }
    c[k] = 3.0 / ((2 * k + 3) as f64 * (k - 1) as f64) * sum;
  }

  let u2 = u * u;
  let mut result = 1.0 / u2;
  let mut u_power = 1.0;
  for k in 1..max_terms {
    u_power *= u2;
    result += c[k] * u_power;
  }
  result
}

/// InverseWeierstrassP[p, {g2, g3}] — inverse Weierstrass elliptic function.
///
/// Returns {u, ℘'(u)} where ℘(u; g₂, g₃) = p.
///
/// InverseWeierstrassP[{p, pp}, {g2, g3}] — returns u given both p and p'.
pub fn inverse_weierstrass_p_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseWeierstrassP expects exactly 2 arguments".into(),
    ));
  }

  let (g2, g3) = match &args[1] {
    Expr::List(items) if items.len() == 2 => (&items[0], &items[1]),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "InverseWeierstrassP".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Try to get g2, g3 as f64
  let g2_f = match try_eval_to_f64(g2) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: "InverseWeierstrassP".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let g3_f = match try_eval_to_f64(g3) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: "InverseWeierstrassP".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Two forms:
  // 1. InverseWeierstrassP[p, {g2, g3}] where p is a number → {u, p'}
  // 2. InverseWeierstrassP[{p, pp}, {g2, g3}] → u
  match &args[0] {
    Expr::List(items) if items.len() == 2 => {
      // Form 2: {p, pp} given
      let p_f = match try_eval_to_f64(&items[0]) {
        Some(v) => v,
        None => {
          return Ok(Expr::FunctionCall {
            name: "InverseWeierstrassP".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let pp_f = match try_eval_to_f64(&items[1]) {
        Some(v) => v,
        None => {
          return Ok(Expr::FunctionCall {
            name: "InverseWeierstrassP".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let u = inverse_weierstrass_p_with_prime(p_f, pp_f, g2_f, g3_f);
      Ok(Expr::Real(u))
    }
    _ => {
      // Form 1: just p given
      let p_f = match try_eval_to_f64(&args[0]) {
        Some(v) => v,
        None => {
          // Check if it's purely symbolic
          if !matches!(&args[0], Expr::Real(_))
            && !matches!(&args[0], Expr::Integer(_))
          {
            return Ok(Expr::FunctionCall {
              name: "InverseWeierstrassP".to_string(),
              args: args.to_vec(),
            });
          }
          return Ok(Expr::FunctionCall {
            name: "InverseWeierstrassP".to_string(),
            args: args.to_vec(),
          });
        }
      };
      // Need at least one Real for numeric eval
      let has_real = matches!(&args[0], Expr::Real(_))
        || matches!(g2, Expr::Real(_))
        || matches!(g3, Expr::Real(_));
      if !has_real {
        return Ok(Expr::FunctionCall {
          name: "InverseWeierstrassP".to_string(),
          args: args.to_vec(),
        });
      }
      let (u, pp) = inverse_weierstrass_p_numeric(p_f, g2_f, g3_f);
      Ok(Expr::List(vec![Expr::Real(u), Expr::Real(pp)]))
    }
  }
}

/// Compute InverseWeierstrassP[p, {g2, g3}] numerically.
/// Returns (u, p') where ℘(u) = p.
/// Uses Newton's method starting from an initial guess.
fn inverse_weierstrass_p_numeric(p: f64, g2: f64, g3: f64) -> (f64, f64) {
  // ℘'² = 4℘³ - g₂℘ - g₃
  let pp_sq = 4.0 * p * p * p - g2 * p - g3;
  let pp = if pp_sq >= 0.0 {
    -pp_sq.sqrt() // Take the negative branch (principal value, small positive u)
  } else {
    // Complex case — for now just use the magnitude
    0.0
  };
  let u = inverse_weierstrass_p_with_prime(p, pp, g2, g3);
  let actual_pp = weierstrass_p_prime_numeric(u, g2, g3);
  (u, actual_pp)
}

/// Compute u given p, p', g2, g3 such that ℘(u) = p and ℘'(u) = pp.
/// Uses Newton's method: u_{n+1} = u_n - (℘(u_n) - p) / ℘'(u_n)
fn inverse_weierstrass_p_with_prime(p: f64, pp: f64, g2: f64, g3: f64) -> f64 {
  // Initial guess: for large p, u ≈ 1/√p (from Laurent series ℘ ≈ 1/u²)
  let mut u = if p.abs() > 1.0 {
    let sign = if pp < 0.0 { 1.0 } else { -1.0 };
    sign / p.abs().sqrt()
  } else {
    // For smaller p, start with a moderate guess
    let sign = if pp < 0.0 { 1.0 } else { -1.0 };
    sign * 0.5
  };

  // Newton's method
  for _ in 0..200 {
    let wp = weierstrass_p_numeric(u, g2, g3);
    let wpp = weierstrass_p_prime_numeric(u, g2, g3);

    if wpp.abs() < 1e-300 {
      break;
    }

    let delta = (wp - p) / wpp;
    u -= delta;

    if delta.abs() < 1e-14 * u.abs().max(1e-14) {
      break;
    }
  }

  u
}

/// ExpIntegralEi[x] - Exponential integral Ei(x)
pub fn exp_integral_ei_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ExpIntegralEi expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    // ExpIntegralEi[0] = -Infinity
    Expr::Integer(0) => Ok(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), Expr::Identifier("Infinity".to_string())],
    }),
    // ExpIntegralEi[Infinity] = Infinity
    Expr::Identifier(s) if s == "Infinity" => {
      Ok(Expr::Identifier("Infinity".to_string()))
    }
    // Numeric evaluation
    Expr::Real(x) => Ok(Expr::Real(exp_integral_ei_numeric(*x))),
    // Check for -Infinity or other cases
    other => {
      if is_neg_infinity(other) {
        return Ok(Expr::Integer(0));
      }
      // Unevaluated
      Ok(Expr::FunctionCall {
        name: "ExpIntegralEi".to_string(),
        args: args.to_vec(),
      })
    }
  }
}

/// Compute Ei(x) numerically
/// For all x: Ei(x) = γ + ln|x| + Σ_{n=1}^∞ x^n / (n * n!)
pub fn exp_integral_ei_numeric(x: f64) -> f64 {
  if x.abs() < 40.0 {
    // Power series: Ei(x) = γ + ln|x| + Σ x^n / (n * n!)
    let euler_gamma = 0.5772156649015329;
    let mut sum = euler_gamma + x.abs().ln();
    let mut term = 1.0;
    for n in 1..200 {
      let nf = n as f64;
      term *= x / nf;
      sum += term / nf;
      if (term / nf).abs() < 1e-16 * sum.abs() {
        break;
      }
    }
    sum
  } else {
    // Asymptotic expansion for large |x|: Ei(x) ~ e^x/x * Σ n!/x^n
    // Use continued fraction for better convergence
    let mut result = 0.0;
    for n in (1..=100).rev() {
      let nf = n as f64;
      result = nf / (1.0 + nf / (x + result));
    }
    result = x.exp() / (x + result);
    if x > 0.0 {
      result
    } else {
      // For large negative x, Ei(x) ≈ e^x/x * (1 + 1!/x + 2!/x^2 + ...)
      let mut sum = 1.0;
      let mut term = 1.0;
      for n in 1..100 {
        term *= n as f64 / x;
        sum += term;
        if term.abs() < 1e-16 * sum.abs() {
          break;
        }
      }
      sum * x.exp() / x
    }
  }
}

/// CosIntegral[z] - Cosine integral Ci(z)
/// Ci(z) = γ + ln(z) + ∫₀ᶻ (cos(t)-1)/t dt
pub fn cos_integral_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "CosIntegral expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    // CosIntegral[0] = -Infinity
    Expr::Integer(0) => Ok(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), Expr::Identifier("Infinity".to_string())],
    }),
    // CosIntegral[Infinity] = 0
    Expr::Identifier(s) if s == "Infinity" => Ok(Expr::Integer(0)),
    // Numeric evaluation
    Expr::Real(x) => Ok(Expr::Real(cos_integral_numeric(*x))),
    // Check for -Infinity
    other => {
      if is_neg_infinity(other) {
        // CosIntegral[-Infinity] = I*Pi
        return Ok(Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Identifier("I".to_string()),
            Expr::Constant("Pi".to_string()),
          ],
        });
      }
      // Unevaluated
      Ok(Expr::FunctionCall {
        name: "CosIntegral".to_string(),
        args: args.to_vec(),
      })
    }
  }
}

/// Compute Ci(z) numerically for real z
/// Ci(z) = γ + ln|z| + ∫₀ᶻ (cos(t)-1)/t dt
/// = γ + ln|z| + Σ_{n=1}^∞ (-1)^n z^(2n) / (2n · (2n)!)
fn cos_integral_numeric(z: f64) -> f64 {
  let euler_gamma = 0.5772156649015329;

  if z.abs() < 40.0 {
    // Power series: Ci(z) = γ + ln|z| + Σ (-1)^n z^(2n) / (2n · (2n)!)
    let mut sum = euler_gamma + z.abs().ln();
    let z2 = z * z;
    let mut term = 1.0; // Will build up z^(2n) / (2n)!
    for n in 1..200 {
      let n2 = (2 * n) as f64;
      // term *= z^2 / ((2n-1) * 2n)
      term *= z2 / ((n2 - 1.0) * n2);
      let contrib = if n % 2 == 1 { -term } else { term };
      sum += contrib / n2;
      if (contrib / n2).abs() < 1e-16 * sum.abs().max(1e-300) {
        break;
      }
    }
    sum
  } else {
    // Asymptotic expansion for large |z|
    // Ci(z) ≈ sin(z)/z * f(z) - cos(z)/z * g(z)
    // where f(z) = Σ (-1)^n (2n)! / z^(2n)
    //       g(z) = Σ (-1)^n (2n+1)! / z^(2n+1)
    let mut f = 0.0;
    let mut g = 0.0;
    let mut f_term = 1.0;
    let mut g_term = 1.0 / z;
    for n in 0..100 {
      f += if n % 2 == 0 { f_term } else { -f_term };
      g += if n % 2 == 0 { g_term } else { -g_term };
      let n2 = (2 * n + 1) as f64;
      let n2p = (2 * n + 2) as f64;
      f_term *= n2 * n2p / (z * z);
      g_term *= n2p * (n2p + 1.0) / (z * z);
      if f_term.abs() < 1e-16 && g_term.abs() < 1e-16 {
        break;
      }
      // Divergent series: stop when terms start growing
      if n > 0
        && (f_term.abs() > (n2 * n2p / (z * z) * f_term).abs()
          || g_term.abs() > 1e10)
      {
        break;
      }
    }
    f / z * z.sin() - g * z.cos()
  }
}

/// SinIntegral[z] - Sine integral Si(z)
/// Si(z) = ∫₀ᶻ sin(t)/t dt
pub fn sin_integral_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "SinIntegral expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    // SinIntegral[0] = 0
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    // SinIntegral[Infinity] = Pi/2
    Expr::Identifier(s) if s == "Infinity" => Ok(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        crate::functions::math_ast::make_rational(1, 2),
        Expr::Constant("Pi".to_string()),
      ],
    }),
    // Numeric evaluation
    Expr::Real(x) => Ok(Expr::Real(sin_integral_numeric(*x))),
    // Check for -Infinity
    other => {
      if is_neg_infinity(other) {
        // SinIntegral[-Infinity] = -Pi/2
        return Ok(Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            crate::functions::math_ast::make_rational(-1, 2),
            Expr::Constant("Pi".to_string()),
          ],
        });
      }
      // Unevaluated
      Ok(Expr::FunctionCall {
        name: "SinIntegral".to_string(),
        args: args.to_vec(),
      })
    }
  }
}

/// Compute Si(z) numerically for real z
/// Si(z) = Σ_{n=0}^∞ (-1)^n z^(2n+1) / ((2n+1) · (2n+1)!)
fn sin_integral_numeric(z: f64) -> f64 {
  if z.abs() < 40.0 {
    let z2 = z * z;
    let mut sum = z;
    let mut term = z;
    for n in 1..200 {
      let n2 = (2 * n) as f64;
      let n2p1 = n2 + 1.0;
      term *= -z2 / (n2 * n2p1);
      sum += term / n2p1;
      if (term / n2p1).abs() < 1e-16 * sum.abs().max(1e-300) {
        break;
      }
    }
    sum
  } else {
    // Asymptotic expansion for large |z|
    // Si(z) ≈ π/2 - cos(z)/z * f(z) - sin(z)/z * g(z)
    let sign = if z > 0.0 { 1.0 } else { -1.0 };
    let az = z.abs();
    let mut f = 0.0;
    let mut g = 0.0;
    let mut f_term = 1.0;
    let mut g_term = 1.0 / az;
    for n in 0..100 {
      f += if n % 2 == 0 { f_term } else { -f_term };
      g += if n % 2 == 0 { g_term } else { -g_term };
      let n2 = (2 * n + 1) as f64;
      let n2p = (2 * n + 2) as f64;
      let old_f = f_term;
      f_term *= n2 * n2p / (az * az);
      g_term *= n2p * (n2p + 1.0) / (az * az);
      if f_term.abs() < 1e-16 && g_term.abs() < 1e-16 {
        break;
      }
      if f_term.abs() > old_f.abs() {
        break;
      }
    }
    sign * (std::f64::consts::FRAC_PI_2 - f / az * az.cos() - g * az.sin())
  }
}

/// ExpIntegralE[n, z] - Generalized exponential integral E_n(z)
pub fn exp_integral_e_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ExpIntegralE expects exactly 2 arguments".into(),
    ));
  }

  let n_expr = &args[0];
  let z_expr = &args[1];

  // E_n(0): E_1(0) = ComplexInfinity, E_n(0) = 1/(n-1) for n > 1
  if is_expr_zero(z_expr)
    && let Expr::Integer(n) = n_expr
  {
    if *n == 1 {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    } else if *n > 1 {
      return Ok(make_rational(1, *n - 1));
    }
  }

  // Numeric evaluation
  let n_val = expr_to_f64(n_expr);
  let z_val = expr_to_f64(z_expr);
  let is_numeric_eval = n_val.is_some()
    && z_val.is_some()
    && (matches!(z_expr, Expr::Real(_)) || matches!(n_expr, Expr::Real(_)));

  if is_numeric_eval {
    let n = n_val.unwrap();
    let z = z_val.unwrap();
    let result = exp_integral_en(n as i64, z);
    return Ok(Expr::Real(result));
  }

  Ok(Expr::FunctionCall {
    name: "ExpIntegralE".to_string(),
    args: args.to_vec(),
  })
}

/// Compute E_n(z) = ∫_1^∞ e^{-zt}/t^n dt
pub fn exp_integral_en(n: i64, z: f64) -> f64 {
  if n == 0 {
    // E_0(z) = e^{-z}/z
    return (-z).exp() / z;
  }

  // Compute E_1(z) first, then use recurrence for higher n
  let e1 = exp_integral_e1(z);

  if n == 1 {
    return e1;
  }

  // Recurrence: E_{n+1}(z) = (e^{-z} - z*E_n(z))/n
  let mut e_prev = e1;
  let exp_neg_z = (-z).exp();
  for k in 1..n {
    let e_next = (exp_neg_z - z * e_prev) / k as f64;
    e_prev = e_next;
  }
  e_prev
}

/// Compute E_1(z) via series for small z, continued fraction for large z
pub fn exp_integral_e1(z: f64) -> f64 {
  let euler_gamma = 0.5772156649015329;

  if z <= 0.0 {
    return f64::INFINITY;
  }

  if z < 1.5 {
    // Series: E_1(z) = -γ - ln(z) + Σ_{n=1}^∞ (-1)^{n+1} z^n / (n * n!)
    let mut sum = -euler_gamma - z.ln();
    let mut term = 1.0;
    for n in 1..200 {
      let nf = n as f64;
      term *= -z / nf;
      sum -= term / nf;
      if (term / nf).abs() < 1e-16 * sum.abs().max(1e-300) {
        break;
      }
    }
    sum
  } else {
    // Continued fraction: E_1(z) = e^{-z} * 1/(z + 1/(1 + 1/(z + 2/(1 + 2/(z + ...)))))
    // Using Lentz-Thompson algorithm with a_i, b_i coefficients
    // E_1(z) = e^{-z} * CF where CF = 1/(z+) 1/(1+) 1/(z+) 2/(1+) 2/(z+) 3/(1+) ...
    // Equivalently using the modified Lentz: a_0=0, b_0=z, then alternating
    // numerators [1, 1, 2, 2, 3, 3, ...] and denominators [1, z, 1, z, 1, z, ...]
    let mut f = z; // b_0
    let mut c = z;
    let mut d = 0.0;

    for i in 1..200 {
      let a = ((i + 1) / 2) as f64; // 1, 1, 2, 2, 3, 3, ...
      let b = if i % 2 == 1 { 1.0 } else { z };

      d = b + a * d;
      if d.abs() < 1e-30 {
        d = 1e-30;
      }
      c = b + a / c;
      if c.abs() < 1e-30 {
        c = 1e-30;
      }
      d = 1.0 / d;
      let delta = c * d;
      f *= delta;
      if (delta - 1.0).abs() < 1e-16 {
        break;
      }
    }

    (-z).exp() / f
  }
}

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
pub fn zeta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Zeta expects exactly 1 argument".into(),
    ));
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

/// LogIntegral[x] - Logarithmic integral Li(x) = Ei(ln(x))
pub fn log_integral_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "LogIntegral expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    // LogIntegral[0] = 0
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    // LogIntegral[1] = -Infinity (pole at x=1 since ln(1)=0)
    Expr::Integer(1) => Ok(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), Expr::Identifier("Infinity".to_string())],
    }),
    // Numeric evaluation: Li(x) = Ei(ln(x))
    Expr::Real(x) => {
      let result = exp_integral_ei_numeric(x.ln());
      Ok(Expr::Real(result))
    }
    // Unevaluated
    _ => Ok(Expr::FunctionCall {
      name: "LogIntegral".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// RiemannR[x] - Riemann's prime counting function estimate
/// Uses the Gram series: R(x) = 1 + Σ_{n=1}^∞ (ln x)^n / (n * n! * ζ(n+1))
pub fn riemann_r_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "RiemannR expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    // RiemannR[1] = 1 (exact special value)
    Expr::Integer(1) => Ok(Expr::Integer(1)),
    // Numeric evaluation for real values
    Expr::Real(x) if *x > 0.0 => Ok(Expr::Real(riemann_r_numeric(*x))),
    // Unevaluated for symbolic and other args
    _ => Ok(Expr::FunctionCall {
      name: "RiemannR".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Compute Riemann R function numerically using the Gram series.
/// Computes (ln x)^n / n! incrementally to avoid precision loss from
/// large intermediate values.  Uses Kahan compensated summation.
fn riemann_r_numeric(x: f64) -> f64 {
  let ln_x = x.ln();
  let mut sum = 1.0_f64;
  let mut comp = 0.0_f64; // Kahan compensation
  let mut ratio = 1.0_f64; // (ln x)^n / n!  (updated incrementally)

  for n in 1..=200 {
    ratio *= ln_x / n as f64; // ratio = (ln x)^n / n!
    let zeta_val = zeta_numeric((n + 1) as f64);
    let term = ratio / (n as f64 * zeta_val);
    // Kahan compensated addition
    let y = term - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
    if term.abs() < 1e-16 * sum.abs() {
      break;
    }
  }
  sum
}

/// HypergeometricPFQ[{a1,...,ap}, {b1,...,bq}, z]
/// Generalized hypergeometric function pFq
pub fn hypergeometric_pfq_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "HypergeometricPFQ expects exactly 3 arguments".into(),
    ));
  }

  let a_list = match &args[0] {
    Expr::List(v) => v.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "HypergeometricPFQ".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let b_list = match &args[1] {
    Expr::List(v) => v.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "HypergeometricPFQ".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let z = &args[2];

  // HypergeometricPFQ[{a...}, {b...}, 0] = 1
  match z {
    Expr::Integer(0) => return Ok(Expr::Integer(1)),
    Expr::Real(x) if *x == 0.0 => return Ok(Expr::Integer(1)),
    _ => {}
  }

  // HypergeometricPFQ[{}, {}, z] = E^z
  if a_list.is_empty() && b_list.is_empty() {
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![Expr::Identifier("E".to_string()), z.clone()],
    });
  }

  // Numeric evaluation: all parameters and z must be numeric
  let z_val = match z {
    Expr::Real(x) => Some(*x),
    Expr::Integer(n) => Some(*n as f64),
    _ => None,
  };
  if z_val.is_none() {
    return Ok(Expr::FunctionCall {
      name: "HypergeometricPFQ".to_string(),
      args: args.to_vec(),
    });
  }
  let z_val = z_val.unwrap();

  let a_vals: Option<Vec<f64>> = a_list
    .iter()
    .map(|e| match e {
      Expr::Real(x) => Some(*x),
      Expr::Integer(n) => Some(*n as f64),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
          Some(*n as f64 / *d as f64)
        } else {
          None
        }
      }
      _ => None,
    })
    .collect();
  let b_vals: Option<Vec<f64>> = b_list
    .iter()
    .map(|e| match e {
      Expr::Real(x) => Some(*x),
      Expr::Integer(n) => Some(*n as f64),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
          Some(*n as f64 / *d as f64)
        } else {
          None
        }
      }
      _ => None,
    })
    .collect();

  if let (Some(a_vals), Some(b_vals)) = (a_vals, b_vals) {
    // Check convergence at |z|=1: for p <= q+1, need sum(b) - sum(a) > 0
    if z_val.abs() >= 1.0 && a_vals.len() <= b_vals.len() + 1 {
      let sum_a: f64 = a_vals.iter().sum();
      let sum_b: f64 = b_vals.iter().sum();
      if z_val.abs() == 1.0 && sum_b - sum_a <= 0.0 {
        return Ok(Expr::Identifier("Infinity".to_string()));
      }
    }
    // For p > q+1 and |z| >= 1, the series diverges
    if z_val.abs() >= 1.0 && a_vals.len() > b_vals.len() + 1 {
      return Ok(Expr::FunctionCall {
        name: "HypergeometricPFQ".to_string(),
        args: args.to_vec(),
      });
    }
    let result = hypergeometric_pfq_numeric(&a_vals, &b_vals, z_val);
    if result.is_infinite() {
      return Ok(Expr::Identifier("Infinity".to_string()));
    }
    return Ok(Expr::Real(result));
  }

  Ok(Expr::FunctionCall {
    name: "HypergeometricPFQ".to_string(),
    args: args.to_vec(),
  })
}

/// Compute generalized hypergeometric function pFq numerically via series
/// Uses Kahan compensated summation for improved precision.
fn hypergeometric_pfq_numeric(a: &[f64], b: &[f64], z: f64) -> f64 {
  let mut sum = 1.0_f64;
  let mut comp = 0.0_f64; // Kahan compensation
  let mut term = 1.0_f64;

  for n in 0..1000 {
    // Multiply by (a1+n)(a2+n)...(ap+n) * z / ((b1+n)(b2+n)...(bq+n) * (n+1))
    let mut num = z;
    for &ai in a {
      num *= ai + n as f64;
    }
    let mut den = (n + 1) as f64;
    for &bi in b {
      let bi_n = bi + n as f64;
      if bi_n == 0.0 {
        return f64::INFINITY; // pole in denominator
      }
      den *= bi_n;
    }
    term *= num / den;
    // Kahan compensated addition
    let y = term - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
    if term.abs() < 1e-16 * sum.abs() {
      break;
    }
    if !sum.is_finite() {
      break;
    }
  }
  sum
}

/// HypergeometricPFQRegularized[{a1,...,ap}, {b1,...,bq}, z]
/// = HypergeometricPFQ[{a1,...,ap}, {b1,...,bq}, z] / (Gamma[b1] * ... * Gamma[bq])
pub fn hypergeometric_pfq_regularized_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "HypergeometricPFQRegularized expects exactly 3 arguments".into(),
    ));
  }

  let b_list = match &args[1] {
    Expr::List(v) => v.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "HypergeometricPFQRegularized".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // If b_list is empty, this is the same as HypergeometricPFQ
  if b_list.is_empty() {
    return hypergeometric_pfq_ast(args);
  }

  // Try to evaluate the underlying HypergeometricPFQ first
  let pfq_result = hypergeometric_pfq_ast(args)?;

  // If it stays symbolic, return regularized form
  match &pfq_result {
    Expr::FunctionCall { name, .. } if name == "HypergeometricPFQ" => {
      return Ok(Expr::FunctionCall {
        name: "HypergeometricPFQRegularized".to_string(),
        args: args.to_vec(),
      });
    }
    _ => {}
  }

  // Handle Infinity result
  if matches!(&pfq_result, Expr::Identifier(s) if s == "Infinity") {
    // Check if any denominator Gamma is infinite (non-positive integer b)
    for b_expr in &b_list {
      if let Some(n) = expr_to_i128(b_expr)
        && n <= 0
      {
        // Gamma has a pole, so regularized form is finite (indeterminate) - return unevaluated
        return Ok(Expr::FunctionCall {
          name: "HypergeometricPFQRegularized".to_string(),
          args: args.to_vec(),
        });
      }
    }
    return Ok(Expr::Identifier("Infinity".to_string()));
  }

  // Check if any argument is Real (to decide numeric vs symbolic evaluation)
  let has_real = matches!(&pfq_result, Expr::Real(_))
    || b_list.iter().any(|e| matches!(e, Expr::Real(_)));

  // Try numeric path: if pfq_result is numeric and all b values yield numeric gamma
  if has_real
    && let Some(pfq_val) = match &pfq_result {
      Expr::Real(x) => Some(*x),
      Expr::Integer(n) => Some(*n as f64),
      Expr::Identifier(s) if s == "Infinity" => Some(f64::INFINITY),
      _ => None,
    }
  {
    // Compute product of Gamma[b_i] numerically
    let b_vals: Option<Vec<f64>> = b_list
      .iter()
      .map(|e| match e {
        Expr::Real(x) => Some(*x),
        Expr::Integer(n) => Some(*n as f64),
        Expr::FunctionCall { name, args }
          if name == "Rational" && args.len() == 2 =>
        {
          if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
            Some(*n as f64 / *d as f64)
          } else {
            None
          }
        }
        _ => None,
      })
      .collect();

    if let Some(b_vals) = b_vals {
      let mut gamma_prod = 1.0_f64;
      for bv in &b_vals {
        gamma_prod *= gamma_fn(*bv);
      }
      if gamma_prod.is_infinite() {
        return Ok(Expr::Integer(0)); // 1/Γ(pole) = 0 for regularization
      }
      let result = pfq_val / gamma_prod;
      if result.is_infinite() {
        return Ok(Expr::Identifier("Infinity".to_string()));
      }
      return Ok(Expr::Real(result));
    }
  }

  // Symbolic path: construct the division
  let mut gamma_product = Expr::Integer(1);
  for b_expr in &b_list {
    let gamma_val = gamma_ast(&[b_expr.clone()])?;
    gamma_product = crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(gamma_product),
      right: Box::new(gamma_val),
    })?;
  }

  crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(pfq_result),
    right: Box::new(gamma_product),
  })
}

/// Hypergeometric2F1Regularized[a, b, c, z] = HypergeometricPFQRegularized[{a,b},{c},z]
pub fn hypergeometric_2f1_regularized_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 4 {
    return Ok(Expr::FunctionCall {
      name: "Hypergeometric2F1Regularized".to_string(),
      args: args.to_vec(),
    });
  }
  let a = args[0].clone();
  let b = args[1].clone();
  let c = args[2].clone();
  let z = args[3].clone();

  let pfq_args = vec![Expr::List(vec![a, b]), Expr::List(vec![c]), z];
  let result = hypergeometric_pfq_regularized_ast(&pfq_args)?;

  // If the result stayed as HypergeometricPFQRegularized, convert back to 2F1 form
  if let Expr::FunctionCall { name, .. } = &result
    && name == "HypergeometricPFQRegularized"
  {
    return Ok(Expr::FunctionCall {
      name: "Hypergeometric2F1Regularized".to_string(),
      args: args.to_vec(),
    });
  }

  Ok(result)
}

/// QPochhammer[a, q, n] — q-Pochhammer symbol.
/// Computes Product[(1 - a*q^k), {k, 0, n-1}] for non-negative integer n.
pub fn q_pochhammer_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "QPochhammer".to_string(),
      args: args.to_vec(),
    });
  }

  let a = &args[0];
  let q = &args[1];
  let n_expr = &args[2];

  // n must be a non-negative integer
  let n = match expr_to_i128(n_expr) {
    Some(n) if n >= 0 => n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "QPochhammer".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // QPochhammer[a, q, 0] = 1
  if n == 0 {
    return Ok(Expr::Integer(1));
  }

  // Compute the product symbolically: Product[(1 - a*q^k), {k, 0, n-1}]
  // Build each factor and multiply using the evaluator
  let mut result = Expr::Integer(1);
  for k in 0..n {
    // Compute q^k
    let qk = if k == 0 {
      Expr::Integer(1)
    } else {
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![q.clone(), Expr::Integer(k as i128)],
      })?
    };
    // Compute a * q^k
    let aqk = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![a.clone(), qk],
    })?;
    // Compute 1 - a*q^k
    let factor =
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          Expr::Integer(1),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), aqk],
          },
        ],
      })?;
    // Multiply into result
    result = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![result, factor],
    })?;
  }

  Ok(result)
}

/// SphericalBesselJ[n, z] — spherical Bessel function of the first kind.
/// SphericalBesselJ[n, z] = Sqrt[Pi/(2z)] * BesselJ[n + 1/2, z]
pub fn spherical_bessel_j_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "SphericalBesselJ".to_string(),
      args: args.to_vec(),
    });
  }

  let n_expr = &args[0];
  let z_expr = &args[1];

  // Handle z = 0 case: SphericalBesselJ[0, 0] = 1, SphericalBesselJ[n, 0] = 0 for n > 0
  if matches!(z_expr, Expr::Integer(0))
    && let Some(n) = expr_to_i128(n_expr)
  {
    if n == 0 {
      return Ok(Expr::Integer(1));
    } else if n > 0 {
      return Ok(Expr::Integer(0));
    }
  }

  // For numeric evaluation, compute Sqrt[Pi/(2z)] * BesselJ[n + 1/2, z]
  let has_real =
    matches!(n_expr, Expr::Real(_)) || matches!(z_expr, Expr::Real(_));
  let n_f64 = try_eval_to_f64(n_expr);
  let z_f64 = try_eval_to_f64(z_expr);

  if has_real && let (Some(n_val), Some(z_val)) = (n_f64, z_f64) {
    let bessel_result =
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "BesselJ".to_string(),
        args: vec![Expr::Real(n_val + 0.5), Expr::Real(z_val)],
      })?;
    if let Some(bj) = try_eval_to_f64(&bessel_result) {
      let result = (std::f64::consts::PI / (2.0 * z_val)).sqrt() * bj;
      return Ok(Expr::Real(result));
    }
  }

  // Return unevaluated for symbolic case
  Ok(Expr::FunctionCall {
    name: "SphericalBesselJ".to_string(),
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

/// PolyLog[s, z] - Polylogarithm function Li_s(z)
pub fn polylog_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "PolyLog expects exactly 2 arguments".into(),
    ));
  }

  let s_expr = &args[0];
  let z_expr = &args[1];

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

/// AiryAi[x] - Airy function of the first kind
pub fn airy_ai_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "AiryAi expects exactly 1 argument".into(),
    ));
  }

  // Numeric evaluation
  if let Some(x_f) = expr_to_f64(&args[0])
    && matches!(&args[0], Expr::Real(_))
  {
    return Ok(Expr::Real(airy_ai(x_f)));
  }

  Ok(Expr::FunctionCall {
    name: "AiryAi".to_string(),
    args: args.to_vec(),
  })
}

/// Compute Ai(x) using the two power series
/// Ai(x) = c1 * f(x) - c2 * g(x)
/// f(x) = 1 + x^3/(2·3) + x^6/(2·3·5·6) + ...
/// g(x) = x + x^4/(3·4) + x^7/(3·4·6·7) + ...
/// c1 = Ai(0), c2 = -Ai'(0)
pub fn airy_ai(x: f64) -> f64 {
  let c1 = 0.3550280538878172; // Ai(0)
  let c2 = 0.2588194037928068; // -Ai'(0)

  // For Ai: use power series for moderate/negative x, asymptotic for large positive x
  // The threshold is lower than Bi because Ai decays exponentially, causing cancellation
  if x < 6.0 {
    let mut f = 1.0;
    let mut g = x;
    let mut f_term = 1.0;
    let mut g_term = x;

    for k in 1..1000 {
      let k3 = 3 * k;
      f_term *= x * x * x / ((k3 as f64 - 1.0) * k3 as f64);
      g_term *= x * x * x / (k3 as f64 * (k3 as f64 + 1.0));
      f += f_term;
      g += g_term;
      if f_term.abs() < 1e-16 * f.abs().max(1e-300)
        && g_term.abs() < 1e-16 * g.abs().max(1e-300)
      {
        break;
      }
    }

    c1 * f - c2 * g
  } else {
    // Asymptotic for large positive x
    let zeta = 2.0 / 3.0 * x.powf(1.5);
    let prefactor =
      (-zeta).exp() / (2.0 * std::f64::consts::PI.sqrt() * x.powf(0.25));
    let mut sum = 1.0;
    let mut term = 1.0;
    let mut prev_term = f64::MAX;
    for k in 1..30 {
      let kf = k as f64;
      let num = (6.0 * kf - 5.0) * (6.0 * kf - 3.0) * (6.0 * kf - 1.0);
      term *= -num / (216.0 * kf * zeta);
      if term.abs() > prev_term {
        break;
      }
      prev_term = term.abs();
      sum += term;
      if term.abs() < 1e-16 {
        break;
      }
    }
    prefactor * sum
  }
}

/// AiryBi[x] - Airy function of the second kind
pub fn airy_bi_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "AiryBi expects exactly 1 argument".into(),
    ));
  }

  // Numeric evaluation
  if let Some(x_f) = expr_to_f64(&args[0])
    && matches!(&args[0], Expr::Real(_))
  {
    return Ok(Expr::Real(airy_bi(x_f)));
  }

  Ok(Expr::FunctionCall {
    name: "AiryBi".to_string(),
    args: args.to_vec(),
  })
}

/// Compute Bi(x) using the two power series
/// Bi(x) = sqrt(3) * (c1 * f(x) + c2 * g(x))
/// where f and g are the same series as for Ai(x)
/// c1 = Ai(0), c2 = -Ai'(0)
pub fn airy_bi(x: f64) -> f64 {
  let c1 = 0.3550280538878172; // Ai(0)
  let c2 = 0.2588194037928068; // -Ai'(0)
  let sqrt3 = 3.0_f64.sqrt();

  if x < 15.0 {
    let mut f = 1.0;
    let mut g = x;
    let mut f_term = 1.0;
    let mut g_term = x;

    for k in 1..1000 {
      let k3 = 3 * k;
      f_term *= x * x * x / ((k3 as f64 - 1.0) * k3 as f64);
      g_term *= x * x * x / (k3 as f64 * (k3 as f64 + 1.0));
      f += f_term;
      g += g_term;
      if f_term.abs() < 1e-16 * f.abs().max(1e-300)
        && g_term.abs() < 1e-16 * g.abs().max(1e-300)
      {
        break;
      }
    }

    sqrt3 * (c1 * f + c2 * g)
  } else {
    // Asymptotic for large positive x
    let zeta = 2.0 / 3.0 * x.powf(1.5);
    let prefactor = zeta.exp() / (std::f64::consts::PI.sqrt() * x.powf(0.25));
    let mut sum = 1.0;
    let mut term = 1.0;
    let mut prev_term = f64::MAX;
    for k in 1..30 {
      let kf = k as f64;
      let num = (6.0 * kf - 5.0) * (6.0 * kf - 3.0) * (6.0 * kf - 1.0);
      term *= num / (216.0 * kf * zeta);
      if term.abs() > prev_term {
        break;
      }
      prev_term = term.abs();
      sum += term;
      if term.abs() < 1e-16 {
        break;
      }
    }
    prefactor * sum
  }
}

/// Hypergeometric1F1[a, b, z] - Kummer's confluent hypergeometric function
pub fn hypergeometric1f1_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "Hypergeometric1F1 expects exactly 3 arguments".into(),
    ));
  }

  // 1F1[a, b, 0] = 1
  if is_expr_zero(&args[2]) {
    return Ok(Expr::Integer(1));
  }

  // Numeric evaluation
  let a_val = expr_to_f64(&args[0]);
  let b_val = expr_to_f64(&args[1]);
  let z_val = expr_to_f64(&args[2]);

  if let (Some(a), Some(b), Some(z)) = (a_val, b_val, z_val) {
    let has_real = matches!(&args[0], Expr::Real(_))
      || matches!(&args[1], Expr::Real(_))
      || matches!(&args[2], Expr::Real(_));
    if has_real {
      return Ok(Expr::Real(hypergeometric_1f1(a, b, z)));
    }
  }

  Ok(Expr::FunctionCall {
    name: "Hypergeometric1F1".to_string(),
    args: args.to_vec(),
  })
}

/// Compute 1F1(a, b; z) = Σ (a)_n z^n / ((b)_n n!)
pub fn hypergeometric_1f1(a: f64, b: f64, z: f64) -> f64 {
  let mut sum = 1.0;
  let mut term = 1.0;

  for n in 0..1000 {
    let nf = n as f64;
    term *= (a + nf) * z / ((b + nf) * (nf + 1.0));
    sum += term;
    if term.abs() < 1e-16 * sum.abs().max(1e-300) {
      break;
    }
  }
  sum
}

/// HypergeometricU[a, b, z] - confluent hypergeometric function of the second kind
pub fn hypergeometric_u_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "HypergeometricU expects exactly 3 arguments".into(),
    ));
  }

  // Numeric evaluation when at least one argument is Real and all are numeric
  let a_val = expr_to_f64(&args[0]);
  let b_val = expr_to_f64(&args[1]);
  let z_val = expr_to_f64(&args[2]);

  if let (Some(a), Some(b), Some(z)) = (a_val, b_val, z_val) {
    let has_real = matches!(&args[0], Expr::Real(_))
      || matches!(&args[1], Expr::Real(_))
      || matches!(&args[2], Expr::Real(_));
    if has_real {
      return Ok(Expr::Real(hypergeometric_u_f64(a, b, z)));
    }
  }

  // Special case: HypergeometricU[a, a+1, z] = z^(-a) (symbolic)
  if let Expr::Integer(a) = &args[0]
    && let Expr::Integer(b) = &args[1]
    && *b == *a + 1
  {
    return Ok(Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![args[2].clone(), Expr::Integer(-*a)],
    });
  }

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "HypergeometricU".to_string(),
    args: args.to_vec(),
  })
}

/// Compute U(a, b, z) numerically using the relation:
/// U(a,b,z) = Γ(1-b)/Γ(a+1-b) * M(a,b,z) + Γ(b-1)/Γ(a) * z^(1-b) * M(a+1-b,2-b,z)
/// where M = 1F1. For integer b, use Richardson extrapolation on the b parameter.
pub fn hypergeometric_u_f64(a: f64, b: f64, z: f64) -> f64 {
  let b_int = b.round();
  let is_b_integer = (b - b_int).abs() < 1e-10;

  if is_b_integer {
    // Use Richardson extrapolation: evaluate at several offsets and extrapolate
    // to the limit b -> integer. This cancels the leading error terms.
    let h = 0.001;
    let u1 = hypergeometric_u_nonint(a, b + h, z);
    let u2 = hypergeometric_u_nonint(a, b - h, z);
    let u3 = hypergeometric_u_nonint(a, b + 2.0 * h, z);
    let u4 = hypergeometric_u_nonint(a, b - 2.0 * h, z);
    // Richardson extrapolation: (4 * f(h) - f(2h)) / 3
    let avg_h = (u1 + u2) / 2.0;
    let avg_2h = (u3 + u4) / 2.0;
    (4.0 * avg_h - avg_2h) / 3.0
  } else {
    hypergeometric_u_nonint(a, b, z)
  }
}

pub fn hypergeometric_u_nonint(a: f64, b: f64, z: f64) -> f64 {
  let g1b = gamma_fn(1.0 - b);
  let ga1b = gamma_fn(a + 1.0 - b);
  let gb1 = gamma_fn(b - 1.0);
  let ga = gamma_fn(a);

  let term1 = if ga1b.is_infinite() || ga1b == 0.0 {
    0.0
  } else {
    g1b / ga1b * hypergeometric_1f1(a, b, z)
  };

  let term2 = if ga.is_infinite() || ga == 0.0 {
    0.0
  } else {
    gb1 / ga * z.powf(1.0 - b) * hypergeometric_1f1(a + 1.0 - b, 2.0 - b, z)
  };

  term1 + term2
}

pub fn hypergeometric2f1_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 4 {
    return Err(InterpreterError::EvaluationError(
      "Hypergeometric2F1 expects exactly 4 arguments".into(),
    ));
  }

  let z = &args[3];

  // Special case: z = 0 => result is 1
  if matches!(z, Expr::Integer(0)) {
    return Ok(Expr::Integer(1));
  }

  // Extract integer values for a, b, c if available
  let a_int = match &args[0] {
    Expr::Integer(n) => Some(*n),
    _ => None,
  };
  let b_int = match &args[1] {
    Expr::Integer(n) => Some(*n),
    _ => None,
  };
  let c_int = match &args[2] {
    Expr::Integer(n) => Some(*n),
    _ => None,
  };

  // a = 0 or b = 0 => 1
  if a_int == Some(0) || b_int == Some(0) {
    return Ok(Expr::Integer(1));
  }

  // 2F1(a, b, b, z) = (1-z)^(-a)
  if crate::syntax::expr_to_string(&args[1])
    == crate::syntax::expr_to_string(&args[2])
  {
    let neg_a = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), args[0].clone()],
    })?;
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::Integer(1),
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(-1), z.clone()],
            },
          ],
        },
        neg_a,
      ],
    });
  }

  // 2F1(a, b, a, z) = (1-z)^(-b)
  if crate::syntax::expr_to_string(&args[0])
    == crate::syntax::expr_to_string(&args[2])
  {
    let neg_b = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), args[1].clone()],
    })?;
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::Integer(1),
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(-1), z.clone()],
            },
          ],
        },
        neg_b,
      ],
    });
  }

  // a is non-positive integer: finite polynomial
  if let Some(a) = a_int
    && a < 0
  {
    return hypergeometric2f1_polynomial(-a as usize, &args[1], &args[2], z);
  }

  // b is non-positive integer: finite polynomial (by symmetry 2F1(a,b,c,z) = 2F1(b,a,c,z))
  if let Some(b) = b_int
    && b < 0
  {
    return hypergeometric2f1_polynomial(-b as usize, &args[0], &args[2], z);
  }

  // 2F1(1, n, n+1, z) for positive integer n: closed form with Log
  if a_int == Some(1)
    && let (Some(b), Some(c)) = (b_int, c_int)
    && b > 0
    && c == b + 1
  {
    return hypergeometric2f1_1_n_np1(b, z);
  }
  // By symmetry: 2F1(n, 1, n+1, z)
  if b_int == Some(1)
    && let (Some(a), Some(c)) = (a_int, c_int)
    && a > 0
    && c == a + 1
  {
    return hypergeometric2f1_1_n_np1(a, z);
  }

  // 2F1(1, b, c, z) for positive integer b < c, c > b+1: partial fraction closed form
  if a_int == Some(1)
    && let (Some(b), Some(c)) = (b_int, c_int)
    && b > 0
    && c > b + 1
  {
    return hypergeometric2f1_1_b_c(b, c, z);
  }
  // By symmetry: 2F1(b, 1, c, z)
  if b_int == Some(1)
    && let (Some(a), Some(c)) = (a_int, c_int)
    && a > 0
    && c > a + 1
  {
    return hypergeometric2f1_1_b_c(a, c, z);
  }

  // Try numeric evaluation when all args are numeric and at least one is Real
  let vals: Vec<Option<f64>> = args
    .iter()
    .map(|a| match a {
      Expr::Integer(n) => Some(*n as f64),
      Expr::Real(f) => Some(*f),
      _ => None,
    })
    .collect();

  let has_real = args.iter().any(|a| matches!(a, Expr::Real(_)));

  if vals.iter().all(|v| v.is_some()) && has_real {
    let a = vals[0].unwrap();
    let b = vals[1].unwrap();
    let c = vals[2].unwrap();
    let z = vals[3].unwrap();
    let result = hypergeometric2f1(a, b, c, z);
    return Ok(Expr::Real(result));
  }

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "Hypergeometric2F1".to_string(),
    args: args.to_vec(),
  })
}

/// Evaluate 2F1(-n, b, c, z) as a finite polynomial (n terms).
/// sum_{k=0}^{n} (-n)_k (b)_k / (c)_k * z^k / k!
fn hypergeometric2f1_polynomial(
  n: usize,
  b: &Expr,
  c: &Expr,
  z: &Expr,
) -> Result<Expr, InterpreterError> {
  // Build the polynomial symbolically: sum_{k=0}^{n} coeff_k * z^k
  // coeff_k = (-n)_k (b)_k / ((c)_k * k!)
  // (-n)_k = (-n)(-n+1)...(-n+k-1) = (-1)^k * n!/(n-k)!
  let mut terms: Vec<Expr> = vec![Expr::Integer(1)]; // k=0 term
  let ni = n as i128;

  for k in 1..=n {
    let ki = k as i128;
    // Build coefficient: product of (-n+j)*(b+j)/(c+j) for j=0..k-1, divided by k!
    // Use symbolic multiplication
    let mut numer_factors: Vec<Expr> = Vec::new();
    let mut denom_factors: Vec<Expr> = Vec::new();

    for j in 0..k {
      let ji = j as i128;
      // (-n+j) factor
      numer_factors.push(Expr::Integer(-ni + ji));
      // (b+j) factor
      numer_factors.push(Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![b.clone(), Expr::Integer(ji)],
      });
      // (c+j) factor in denominator
      denom_factors.push(Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![c.clone(), Expr::Integer(ji)],
      });
    }
    // k! in denominator
    denom_factors.push(Expr::Integer(factorial_i128(k).unwrap_or(1)));

    // z^k
    let zk = if k == 1 {
      z.clone()
    } else {
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![z.clone(), Expr::Integer(ki)],
      }
    };
    numer_factors.push(zk);

    let numer = Expr::FunctionCall {
      name: "Times".to_string(),
      args: numer_factors,
    };
    let denom = Expr::FunctionCall {
      name: "Times".to_string(),
      args: denom_factors,
    };
    terms.push(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        numer,
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![denom, Expr::Integer(-1)],
        },
      ],
    });
  }

  let sum = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms,
  };
  crate::evaluator::evaluate_expr_to_expr(&sum)
}

/// Evaluate 2F1(1, n, n+1, z) for positive integer n.
/// Result: -(n/z^n) * (sum_{k=1}^{n-1} z^k/k + Log[1-z])
fn hypergeometric2f1_1_n_np1(
  n: i128,
  z: &Expr,
) -> Result<Expr, InterpreterError> {
  // Build: -(n/z^n) * (sum_{k=1}^{n-1} z^k/k + Log[1-z])
  let one_minus_z = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      Expr::Integer(1),
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), z.clone()],
      },
    ],
  };
  let log_1mz = Expr::FunctionCall {
    name: "Log".to_string(),
    args: vec![one_minus_z],
  };

  // Build the polynomial sum: sum_{k=1}^{n-1} z^k / k
  let mut inner_terms: Vec<Expr> = Vec::new();
  for k in 1..n {
    let zk = if k == 1 {
      z.clone()
    } else {
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![z.clone(), Expr::Integer(k)],
      }
    };
    if k == 1 {
      inner_terms.push(zk);
    } else {
      inner_terms.push(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(1), Expr::Integer(k)],
          },
          zk,
        ],
      });
    }
  }
  inner_terms.push(log_1mz);

  let inner = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: inner_terms,
  };

  // -(n/z^n) * inner = Times[-n, Power[z, -n], inner]
  let result = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      Expr::Integer(-n),
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![z.clone(), Expr::Integer(-n)],
      },
      inner,
    ],
  };

  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Evaluate 2F1(1, b, c, z) for positive integers b, c with c > b + 1.
/// Uses partial fraction decomposition of (b)_k/(c)_k.
///
/// The series 2F1(1,b,c,z) = sum_{k=0}^inf (b)_k/(c)_k * z^k is decomposed via
/// partial fractions into a sum involving Log(1-z) and polynomial terms, then
/// factored into the canonical form that matches Wolfram's output.
fn hypergeometric2f1_1_b_c(
  b: i128,
  c: i128,
  z: &Expr,
) -> Result<Expr, InterpreterError> {
  use crate::functions::math_ast::numeric_utils::gcd;
  use std::collections::BTreeMap;

  let m = c - b; // >= 2

  // Compute prefactor = (c-1)! / (b-1)! = b * (b+1) * ... * (c-1)
  let mut prefactor: i128 = 1;
  for i in b..c {
    prefactor *= i;
  }

  // Compute C_j = prefactor * (-1)^j / (j! * (m-1-j)!) for j = 0..m-1
  let mut cj: Vec<(i128, i128)> = Vec::new(); // (numerator, denominator)
  for j in 0..m {
    let sign: i128 = if j % 2 == 0 { 1 } else { -1 };
    let mut j_fact: i128 = 1;
    for k in 1..=j {
      j_fact *= k;
    }
    let mut mj_fact: i128 = 1;
    for k in 1..=(m - 1 - j) {
      mj_fact *= k;
    }
    let n = sign * prefactor;
    let d = j_fact * mj_fact;
    let g = gcd(n.abs(), d);
    cj.push((n / g, d / g));
  }

  // Collect all distributed terms into (has_log, z_power) -> (num, den)
  // BTreeMap orders (false, _) before (true, _), and by z_power ascending,
  // which matches Wolfram's canonical Plus ordering.
  let mut collected: BTreeMap<(bool, i128), (i128, i128)> = BTreeMap::new();

  fn add_rational(
    map: &mut BTreeMap<(bool, i128), (i128, i128)>,
    key: (bool, i128),
    num: i128,
    den: i128,
  ) {
    use crate::functions::math_ast::numeric_utils::gcd;
    let entry = map.entry(key).or_insert((0, 1));
    let new_num = entry.0 * den + num * entry.1;
    let new_den = entry.1 * den;
    if new_num == 0 {
      *entry = (0, 1);
    } else {
      let g = gcd(new_num.abs(), new_den.abs());
      *entry = (new_num / g, new_den / g);
    }
  }

  for j in 0..m {
    let (cn, cd) = cj[j as usize];

    // Log term: -C_j * z^{m-1-j} * Log[1-z]
    add_rational(&mut collected, (true, m - 1 - j), -cn, cd);

    // Poly terms: -C_j/i * z^{m-1-j+i} for i = 1..b+j-1
    for i in 1..(b + j) {
      let num = -cn;
      let den = cd * i;
      let g = gcd(num.abs(), den.abs());
      add_rational(&mut collected, (false, m - 1 - j + i), num / g, den / g);
    }
  }

  // Remove zero entries
  collected.retain(|_, (n, _)| *n != 0);

  // Find common denominator across all terms
  let common_den: i128 = collected.values().fold(1i128, |acc, &(_, d)| {
    let g = gcd(acc, d.abs());
    acc / g * d.abs()
  });

  // Scale all numerators to common denominator
  let scaled: Vec<((bool, i128), i128)> = collected
    .iter()
    .map(|(&key, &(n, d))| (key, n * (common_den / d)))
    .collect();

  // Find GCD of all scaled numerators
  let num_gcd = scaled.iter().map(|(_, n)| n.abs()).fold(0i128, gcd);

  if num_gcd == 0 {
    return Ok(Expr::Integer(0));
  }

  // Sign convention: make the coefficient of the highest z-power polynomial term positive
  let max_poly_entry = scaled
    .iter()
    .filter(|((has_log, _), _)| !has_log)
    .max_by_key(|((_, power), _)| *power);

  let sign_adjust: i128 = match max_poly_entry {
    Some((_, coeff)) if *coeff < 0 => -1,
    _ => 1,
  };

  // Overall factor = sign_adjust * num_gcd / common_den
  let factor_num = sign_adjust * num_gcd;
  let factor_den = common_den;
  let fg = gcd(factor_num.abs(), factor_den);
  let (factor_n, factor_d) = (factor_num / fg, factor_den / fg);

  // Build Log[1-z]
  let one_minus_z = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      Expr::Integer(1),
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), z.clone()],
      },
    ],
  };
  let log_1mz = Expr::FunctionCall {
    name: "Log".to_string(),
    args: vec![one_minus_z],
  };

  // Build the Plus terms with factored coefficients
  let mut plus_terms: Vec<Expr> = Vec::new();

  for &((has_log, power), scaled_num) in &scaled {
    // Factored coefficient: scaled_num / (sign_adjust * num_gcd)
    let cn = scaled_num * sign_adjust;
    let cd = num_gcd;
    let cg = gcd(cn.abs(), cd);
    let (cn, cd) = (cn / cg, cd / cg);

    let mut factors: Vec<Expr> = Vec::new();

    // Add coefficient (skip if coefficient is 1)
    if cn == -1 && cd == 1 {
      factors.push(Expr::Integer(-1));
    } else if !(cn == 1 && cd == 1) {
      if cd == 1 {
        factors.push(Expr::Integer(cn));
      } else {
        factors.push(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(cn), Expr::Integer(cd)],
        });
      }
    }

    // Add z^power
    if power == 1 {
      factors.push(z.clone());
    } else if power > 1 {
      factors.push(Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![z.clone(), Expr::Integer(power)],
      });
    }

    // Add Log[1-z] if this is a log term
    if has_log {
      factors.push(log_1mz.clone());
    }

    let term = if factors.is_empty() {
      Expr::Integer(cn) // constant term
    } else if factors.len() == 1 {
      factors.pop().unwrap()
    } else {
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: factors,
      }
    };

    plus_terms.push(term);
  }

  let inner = if plus_terms.len() == 1 {
    plus_terms.pop().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: plus_terms,
    }
  };

  // Build: factor * Power[z, -(c-1)] * inner
  let mut outer_factors: Vec<Expr> = Vec::new();

  if factor_d == 1 {
    if factor_n != 1 {
      outer_factors.push(Expr::Integer(factor_n));
    }
  } else {
    outer_factors.push(Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(factor_n), Expr::Integer(factor_d)],
    });
  }

  outer_factors.push(Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![z.clone(), Expr::Integer(-(c - 1))],
  });

  outer_factors.push(inner);

  let result = if outer_factors.len() == 1 {
    outer_factors.pop().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: outer_factors,
    }
  };

  // Evaluate to get canonical form
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Compute 2F1(a, b; c; z) using series expansion
pub fn hypergeometric2f1(a: f64, b: f64, c: f64, z: f64) -> f64 {
  // Series: sum_{n=0}^{inf} (a)_n (b)_n / (c)_n / n! * z^n
  let mut sum = 1.0;
  let mut term = 1.0;

  for n in 0..1000 {
    let nf = n as f64;
    term *= (a + nf) * (b + nf) / (c + nf) * z / (nf + 1.0);
    sum += term;
    if term.abs() < 1e-16 * sum.abs().max(1e-300) {
      break;
    }
  }
  sum
}

/// ProductLog[z] - Lambert W function (principal branch)
pub fn product_log_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ProductLog expects 1 argument".into(),
    ));
  }
  match &args[0] {
    // ProductLog[0] = 0
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    // ProductLog[E] = 1
    Expr::Identifier(s) if s == "E" => return Ok(Expr::Integer(1)),
    Expr::Constant(s) if s == "E" => return Ok(Expr::Integer(1)),
    // ProductLog[-1/E] = -1
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left,
      right,
    } => {
      if let Expr::Integer(-1) = left.as_ref()
        && (matches!(right.as_ref(), Expr::Identifier(s) if s == "E")
          || matches!(right.as_ref(), Expr::Constant(s) if s == "E"))
      {
        return Ok(Expr::Integer(-1));
      }
    }
    // ProductLog[x.] for float
    Expr::Real(f) => {
      if *f >= -1.0 / std::f64::consts::E {
        // Use iterative approximation (Halley's method)
        let x = *f;
        let mut w = if x < 1.0 { x } else { x.ln() };
        for _ in 0..50 {
          let ew = w.exp();
          let wew = w * ew;
          let delta = wew - x;
          if delta.abs() < 1e-15 {
            break;
          }
          w -= delta / (ew * (w + 1.0) - (w + 2.0) * delta / (2.0 * (w + 1.0)));
        }
        return Ok(Expr::Real(w));
      }
    }
    _ => {}
  }
  Ok(Expr::FunctionCall {
    name: "ProductLog".to_string(),
    args: args.to_vec(),
  })
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

/// Helper: convert a list of Expr to Vec<f64>, returning None if any element is not numeric
fn expr_list_to_f64_vec(list: &[Expr]) -> Option<Vec<f64>> {
  list
    .iter()
    .map(|e| match e {
      Expr::Real(x) => Some(*x),
      Expr::Integer(n) => Some(*n as f64),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
          Some(*n as f64 / *d as f64)
        } else {
          None
        }
      }
      _ => None,
    })
    .collect()
}

/// MeijerG[{{a1,...,an}, {an+1,...,ap}}, {{b1,...,bm}, {bm+1,...,bq}}, z]
/// Meijer G-function
pub fn meijer_g_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "MeijerG".to_string(),
      args: args.to_vec(),
    });
  }

  // Parse upper parameters: {{a1,...,an}, {an+1,...,ap}}
  let (upper_n, upper_rest) = match &args[0] {
    Expr::List(v) if v.len() == 2 => {
      let list_n = match &v[0] {
        Expr::List(l) => l.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "MeijerG".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let list_rest = match &v[1] {
        Expr::List(l) => l.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "MeijerG".to_string(),
            args: args.to_vec(),
          });
        }
      };
      (list_n, list_rest)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MeijerG".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Parse lower parameters: {{b1,...,bm}, {bm+1,...,bq}}
  let (lower_m, lower_rest) = match &args[1] {
    Expr::List(v) if v.len() == 2 => {
      let list_m = match &v[0] {
        Expr::List(l) => l.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "MeijerG".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let list_rest = match &v[1] {
        Expr::List(l) => l.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "MeijerG".to_string(),
            args: args.to_vec(),
          });
        }
      };
      (list_m, list_rest)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MeijerG".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let z = &args[2];

  let n = upper_n.len(); // number of a parameters in first list
  let _p = n + upper_rest.len(); // total number of upper parameters
  let m = lower_m.len(); // number of b parameters in first list
  let _q = m + lower_rest.len(); // total number of lower parameters

  // Consistency check: need m > 0 or n > 0, and p+q < 2(m+n) or other conditions
  if m == 0 && n == 0 {
    // No poles to sum over - function doesn't exist
    return Ok(Expr::FunctionCall {
      name: "MeijerG".to_string(),
      args: args.to_vec(),
    });
  }

  // MeijerG[{{}, {}}, {{0}, {}}, 0] = 1
  if let Expr::Integer(0) = z
    && n == 0
    && m == 1
    && lower_rest.is_empty()
    && upper_n.is_empty()
    && upper_rest.is_empty()
    && let Some(b0) = expr_to_i128(&lower_m[0])
    && b0 == 0
  {
    return Ok(Expr::Integer(1));
  }

  // Try numeric evaluation
  let z_val = match z {
    Expr::Real(x) => Some(*x),
    Expr::Integer(n) => Some(*n as f64),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some(*n as f64 / *d as f64)
      } else {
        None
      }
    }
    _ => None,
  };

  let has_real = matches!(z, Expr::Real(_))
    || upper_n.iter().any(|e| matches!(e, Expr::Real(_)))
    || upper_rest.iter().any(|e| matches!(e, Expr::Real(_)))
    || lower_m.iter().any(|e| matches!(e, Expr::Real(_)))
    || lower_rest.iter().any(|e| matches!(e, Expr::Real(_)));

  if let Some(z_val) = z_val {
    let a_n_vals = expr_list_to_f64_vec(&upper_n);
    let a_rest_vals = expr_list_to_f64_vec(&upper_rest);
    let b_m_vals = expr_list_to_f64_vec(&lower_m);
    let b_rest_vals = expr_list_to_f64_vec(&lower_rest);

    if let (Some(a_n), Some(a_rest), Some(b_m), Some(b_rest)) =
      (a_n_vals, a_rest_vals, b_m_vals, b_rest_vals)
    {
      // Check hdiv condition: a_i - b_j must not be a positive integer
      // for i=1,...,n and j=1,...,m
      for &ai in &a_n {
        for &bj in &b_m {
          let diff = ai - bj;
          if diff > 0.0
            && (diff - diff.round()).abs() < 1e-14
            && diff.round() >= 1.0
          {
            // hdiv: function does not exist
            return Ok(Expr::FunctionCall {
              name: "MeijerG".to_string(),
              args: args.to_vec(),
            });
          }
        }
      }

      // All parameters are numeric - compute
      let mut all_a = a_n.clone();
      all_a.extend_from_slice(&a_rest);
      let mut all_b = b_m.clone();
      all_b.extend_from_slice(&b_rest);

      if has_real || matches!(z, Expr::Real(_)) {
        let result = meijer_g_numeric(n, m, &all_a, &all_b, z_val);
        if result.is_finite() {
          return Ok(Expr::Real(result));
        }
      } else {
        // Integer/rational args - only evaluate via N[]
        // Return unevaluated for pure integer/rational input
        return Ok(Expr::FunctionCall {
          name: "MeijerG".to_string(),
          args: args.to_vec(),
        });
      }
    }
  }

  Ok(Expr::FunctionCall {
    name: "MeijerG".to_string(),
    args: args.to_vec(),
  })
}

/// Numeric evaluation of MeijerG using residue series.
///
/// G^{m,n}_{p,q}(z | a_1,...,a_p; b_1,...,b_q) =
///   -Σ Res[integrand, s = b_h + k] for h=0..m-1, k=0,1,2,...
fn meijer_g_numeric(n: usize, m: usize, a: &[f64], b: &[f64], z: f64) -> f64 {
  let p = a.len();
  let q = b.len();

  // z = 0 special case
  if z == 0.0 {
    if m == 1 && b[0] == 0.0 && n == 0 && p == 0 {
      return 1.0;
    }
    return f64::NAN;
  }

  // Choose convergent series:
  // Left series (poles of Γ(b_j-s)) converges when q > p, or q = p and |z| < 1
  // Right series (poles of Γ(1-a_j+s)) converges when p > q, or p = q and |z| > 1
  // For right series, use the inversion formula: G^{m,n}_{p,q}(z) = G^{n,m}_{q,p}(1/z | ...)
  let use_left = if q > p {
    true
  } else if p > q {
    false
  } else {
    // p == q
    z.abs() <= 1.0
  };

  if use_left {
    meijer_g_direct_series(n, m, p, q, a, b, z)
  } else {
    // Apply inversion: G^{m,n}_{p,q}(z | a; b) = G^{n,m}_{q,p}(1/z | 1-b; 1-a)
    // The parameter order is preserved: new upper = {1-b_1,...,1-b_q}
    //                                   new lower = {1-a_1,...,1-a_p}
    // with new_n = m (first m upper params), new_m = n (first n lower params)
    let new_a: Vec<f64> = b.iter().map(|&bi| 1.0 - bi).collect();
    let new_b: Vec<f64> = a.iter().map(|&ai| 1.0 - ai).collect();
    let new_n = m;
    let new_m = n;
    let new_p = q;
    let new_q = p;
    meijer_g_direct_series(new_n, new_m, new_p, new_q, &new_a, &new_b, 1.0 / z)
  }
}

/// Direct residue series computation for MeijerG.
/// Computes residues numerically at each pole location, handling
/// coinciding poles and zero-pole cancellations automatically.
fn meijer_g_direct_series(
  n: usize,
  m: usize,
  p: usize,
  q: usize,
  a: &[f64],
  b: &[f64],
  z: f64,
) -> f64 {
  let max_terms = 500;

  // Evaluate the full MeijerG integrand at a point s (away from poles):
  // I(s) = ∏_{j<m} Γ(b_j-s) * ∏_{j<n} Γ(1-a_j+s) / ∏_{j≥n,j<p} Γ(a_j-s) / ∏_{j≥m,j<q} Γ(1-b_j+s) * z^s
  let eval_integrand = |s: f64| -> f64 {
    let mut val = z.powf(s);
    for j in 0..m {
      val *= gamma_fn(b[j] - s);
    }
    for j in 0..n {
      val *= gamma_fn(1.0 - a[j] + s);
    }
    for j in n..p {
      let g = gamma_fn(a[j] - s);
      if g.abs() < 1e-300 {
        return 0.0;
      }
      val /= g;
    }
    for j in m..q {
      let g = gamma_fn(1.0 - b[j] + s);
      if g.abs() < 1e-300 {
        return 0.0;
      }
      val /= g;
    }
    val
  };

  let mut total = 0.0;

  for h in 0..m {
    for k in 0..max_terms {
      let s0 = b[h] + k as f64;

      // Check if this pole location is already "owned" by a smaller h
      let mut already_counted = false;
      for j in 0..h {
        let diff = s0 - b[j];
        if diff >= -1e-14
          && (diff - diff.round()).abs() < 1e-10
          && diff.round() >= 0.0
        {
          already_counted = true;
          break;
        }
      }
      if already_counted {
        continue;
      }

      // Count apparent pole order from numerator Gamma functions
      let mut pole_order: usize = 0;
      for j in 0..m {
        let diff = s0 - b[j];
        if diff >= -1e-14 && (diff - diff.round()).abs() < 1e-10 {
          pole_order += 1;
        }
      }
      for j in 0..n {
        let arg = 1.0 - a[j] + s0;
        if arg <= 1e-14
          && (arg - arg.round()).abs() < 1e-10
          && arg.round() <= 0.0
        {
          pole_order += 1;
        }
      }

      // Count zeros from denominator 1/Γ functions
      let mut zero_order = 0;
      for j in n..p {
        let arg = a[j] - s0;
        if (arg - arg.round()).abs() < 1e-10 && arg.round() <= 0.0 {
          zero_order += 1;
        }
      }
      for j in m..q {
        let arg = 1.0 - b[j] + s0;
        if (arg - arg.round()).abs() < 1e-10 && arg.round() <= 0.0 {
          zero_order += 1;
        }
      }

      let effective_order = pole_order.saturating_sub(zero_order);

      if effective_order == 0 {
        continue; // no pole here
      }

      // Compute residue numerically using:
      // g(s) = (s-s₀)^{pole_order} * I(s)  [regularized integrand]
      // The effective pole is of order effective_order in g.
      // Residue of I at s₀ = g^{(pole_order-1)}(s₀) / (pole_order-1)!
      let res = meijer_g_numerical_residue(&eval_integrand, s0, pole_order);

      if !res.is_finite() || res.is_nan() {
        continue;
      }

      let prev_total = total;
      total -= res; // G = -Σ Res

      // Convergence check
      if k > 5 && res.abs() < 1e-14 * total.abs().max(1e-100) {
        break;
      }
      if k > 5
        && prev_total != 0.0
        && (total - prev_total).abs() < 1e-14 * total.abs().max(1e-100)
      {
        break;
      }
    }
  }

  total
}

/// Compute residue of f(s) at a pole of given apparent order using numerical differentiation.
/// Uses the regularized function g(s) = (s-s₀)^order * f(s).
/// Residue = g^{(order-1)}(s₀) / (order-1)!
///
/// IMPORTANT: We never evaluate g(s) at exactly s₀ because of 0*∞ issues.
/// Instead we use symmetric sample points offset from s₀.
fn meijer_g_numerical_residue(
  f: &dyn Fn(f64) -> f64,
  s0: f64,
  order: usize,
) -> f64 {
  let delta = 1e-4;

  let eval_g = |s: f64| -> f64 {
    let eps = s - s0;
    eps.powi(order as i32) * f(s)
  };

  let factorial = |n: usize| -> f64 {
    let mut fac = 1.0;
    for i in 2..=n {
      fac *= i as f64;
    }
    fac
  };

  if order == 1 {
    // Residue = g(s₀) = lim_{ε→0} ε * f(s₀+ε)
    // Use multiple points for Richardson extrapolation
    let g1 = eval_g(s0 + delta);
    let g2 = eval_g(s0 + delta / 2.0);
    let g3 = eval_g(s0 + delta / 4.0);
    // g should converge to the residue as δ→0
    // Use Richardson: 2*g2 - g1 (if linear error)
    let r1 = 2.0 * g2 - g1;
    let r2 = 2.0 * g3 - g2;
    // Second level Richardson
    let result = (4.0 * r2 - r1) / 3.0;
    if result.is_finite() { result } else { g3 }
  } else if order == 2 {
    // First derivative of g at s₀ using central difference (avoiding s₀)
    // g'(s₀) ≈ [g(s₀+δ) - g(s₀-δ)] / (2δ)
    // Use Richardson extrapolation for better accuracy
    let d1 = (eval_g(s0 + delta) - eval_g(s0 - delta)) / (2.0 * delta);
    let d2 = (eval_g(s0 + delta / 2.0) - eval_g(s0 - delta / 2.0)) / delta;
    // Richardson: (4*d2 - d1) / 3
    let result = (4.0 * d2 - d1) / 3.0;
    if result.is_finite() { result } else { d2 }
  } else {
    // Higher order: compute (order-1)-th derivative using finite differences
    // Use half-step offsets to avoid sampling at s₀ (where 0*∞ issues occur)
    let n_pts = order + 4;
    let values: Vec<f64> = (0..n_pts)
      .map(|i| {
        let s = s0 + (i as f64 - (n_pts as f64 - 1.0) / 2.0 + 0.5) * delta;
        eval_g(s)
      })
      .collect();

    let target_deriv = order - 1;
    let mut diff = values;
    for _ in 0..target_deriv {
      let new_len = diff.len() - 1;
      diff = (0..new_len)
        .map(|i| (diff[i + 1] - diff[i]) / delta)
        .collect();
    }

    let center = diff.len() / 2;
    diff[center] / factorial(target_deriv)
  }
}

/// StruveH[n, z] — Struve function H_n(z).
///
/// Series: H_n(z) = sum_{m=0}^{inf} (-1)^m / (Gamma(m + 3/2) * Gamma(m + n + 3/2)) * (z/2)^(2m+n+1)
pub fn struve_h_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StruveH expects exactly 2 arguments".into(),
    ));
  }
  let n_expr = &args[0];
  let z_expr = &args[1];

  // Extract numeric values
  let n_val = match n_expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  };
  let z_val = match z_expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  };

  // Special case: StruveH[n, 0] for non-negative integer n => 0
  if matches!(z_expr, Expr::Integer(0))
    && let Expr::Integer(n) = n_expr
    && *n >= 0
  {
    return Ok(Expr::Integer(0));
  }

  // Numeric evaluation when both args are numeric and at least one is Real
  let is_numeric_eval = n_val.is_some()
    && z_val.is_some()
    && (matches!(z_expr, Expr::Real(_)) || matches!(n_expr, Expr::Real(_)));

  if is_numeric_eval {
    let n = n_val.unwrap();
    let z = z_val.unwrap();
    let result = struve_h(n, z);
    return Ok(Expr::Real(result));
  }

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "StruveH".to_string(),
    args: args.to_vec(),
  })
}

/// Compute Struve H_n(z) using series expansion.
///
/// H_n(z) = sum_{m=0}^{inf} (-1)^m / (Gamma(m + 3/2) * Gamma(m + n + 3/2)) * (z/2)^(2m+n+1)
pub fn struve_h(n: f64, z: f64) -> f64 {
  // Special case: z = 0
  if z == 0.0 {
    if n >= -1.0 {
      return 0.0;
    }
    // For n < -1 with z=0, it may be divergent
    return f64::NAN;
  }

  let half_z = z / 2.0;

  // Series expansion
  let mut sum = 0.0;
  let gamma_3_2 = gamma_fn(1.5); // Gamma(3/2) = sqrt(pi)/2
  let first_gamma_denom = gamma_fn(n + 1.5);

  // First term (m=0): (z/2)^(n+1) / (Gamma(3/2) * Gamma(n + 3/2))
  let mut term = half_z.powf(n + 1.0) / (gamma_3_2 * first_gamma_denom);
  sum += term;

  for m in 1..300 {
    // Ratio of consecutive terms:
    // term_m / term_{m-1} = -half_z^2 / ((m + 0.5) * (m + n + 0.5))
    term *= -half_z * half_z / ((m as f64 + 0.5) * (m as f64 + n + 0.5));
    sum += term;
    if term.abs() < 1e-17 * sum.abs().max(1e-300) {
      break;
    }
  }

  sum
}

/// StruveL[n, z] — Modified Struve function L_n(z).
///
/// Series: L_n(z) = sum_{m=0}^{inf} 1 / (Gamma(m + 3/2) * Gamma(m + n + 3/2)) * (z/2)^(2m+n+1)
pub fn struve_l_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StruveL expects exactly 2 arguments".into(),
    ));
  }
  let n_expr = &args[0];
  let z_expr = &args[1];

  // Extract numeric values
  let n_val = match n_expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  };
  let z_val = match z_expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  };

  // Special case: StruveL[n, 0] for non-negative integer n => 0
  if matches!(z_expr, Expr::Integer(0))
    && let Expr::Integer(n) = n_expr
    && *n >= 0
  {
    return Ok(Expr::Integer(0));
  }

  // Numeric evaluation when both args are numeric and at least one is Real
  let is_numeric_eval = n_val.is_some()
    && z_val.is_some()
    && (matches!(z_expr, Expr::Real(_)) || matches!(n_expr, Expr::Real(_)));

  if is_numeric_eval {
    let n = n_val.unwrap();
    let z = z_val.unwrap();
    let result = struve_l(n, z);
    return Ok(Expr::Real(result));
  }

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "StruveL".to_string(),
    args: args.to_vec(),
  })
}

/// Compute modified Struve L_n(z) using series expansion.
///
/// L_n(z) = sum_{m=0}^{inf} 1 / (Gamma(m + 3/2) * Gamma(m + n + 3/2)) * (z/2)^(2m+n+1)
pub fn struve_l(n: f64, z: f64) -> f64 {
  // Special case: z = 0
  if z == 0.0 {
    if n >= -1.0 {
      return 0.0;
    }
    // For n < -1 with z=0, it may be divergent
    return f64::NAN;
  }

  let half_z = z / 2.0;

  // Series expansion
  let mut sum = 0.0;
  let gamma_3_2 = gamma_fn(1.5); // Gamma(3/2) = sqrt(pi)/2
  let first_gamma_denom = gamma_fn(n + 1.5);

  // First term (m=0): (z/2)^(n+1) / (Gamma(3/2) * Gamma(n + 3/2))
  let mut term = half_z.powf(n + 1.0) / (gamma_3_2 * first_gamma_denom);
  sum += term;

  for m in 1..300 {
    // Ratio of consecutive terms (no alternating sign, unlike StruveH):
    // term_m / term_{m-1} = half_z^2 / ((m + 0.5) * (m + n + 0.5))
    term *= half_z * half_z / ((m as f64 + 0.5) * (m as f64 + n + 0.5));
    sum += term;
    if term.abs() < 1e-17 * sum.abs().max(1e-300) {
      break;
    }
  }

  sum
}

/// SquareWave[t] - square wave with period 1: +1 for frac(t) in [0,1/2), -1 for [1/2,1)
/// SquareWave[{d1, d2, ...}, t] - generalized multi-level square wave
pub fn square_wave_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match args.len() {
    1 => {
      // SquareWave[t]
      if let Some(t) = expr_to_f64(&args[0]) {
        let frac = t - t.floor();
        if frac < 0.5 {
          return Ok(Expr::Integer(1));
        } else {
          return Ok(Expr::Integer(-1));
        }
      }
      // Exact rational input
      if let Expr::FunctionCall {
        name,
        args: rat_args,
      } = &args[0]
        && name == "Rational"
        && rat_args.len() == 2
        && let (Expr::Integer(n), Expr::Integer(d)) =
          (&rat_args[0], &rat_args[1])
      {
        let rem = n.rem_euclid(*d);
        // frac = rem/d, compare with 1/2 => 2*rem < d
        if 2 * rem < *d {
          return Ok(Expr::Integer(1));
        } else {
          return Ok(Expr::Integer(-1));
        }
      }
      Ok(Expr::FunctionCall {
        name: "SquareWave".to_string(),
        args: args.to_vec(),
      })
    }
    2 => {
      // SquareWave[{d1, d2, ...}, t]
      if let Expr::List(levels) = &args[0]
        && let Some(t) = expr_to_f64(&args[1])
      {
        let n = levels.len();
        if n == 0 {
          return Ok(Expr::Integer(0));
        }
        let frac = t - t.floor();
        let idx_raw = (frac * n as f64).floor() as usize;
        let idx = (n - 1).saturating_sub(idx_raw.min(n - 1));
        return Ok(levels[idx].clone());
      }
      Ok(Expr::FunctionCall {
        name: "SquareWave".to_string(),
        args: args.to_vec(),
      })
    }
    _ => Err(InterpreterError::EvaluationError(
      "SquareWave expects 1 or 2 arguments".into(),
    )),
  }
}

/// TriangleWave[t] - triangle wave with period 1: linearly goes from 0 to 1 at t=1/4,
/// back to 0 at t=1/2, down to -1 at t=3/4, and back to 0 at t=1.
/// Formula: 4 * |frac(t + 3/4) - 1/2| - 1
/// TriangleWave[{min, max}, t] - scales output from [-1,1] to [min,max]
pub fn triangle_wave_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match args.len() {
    1 => {
      // Integer input: TriangleWave[n] = 0 for all integers
      if let Expr::Integer(_) = &args[0] {
        return Ok(Expr::Integer(0));
      }
      // Exact rational input
      if let Expr::FunctionCall {
        name,
        args: rat_args,
      } = &args[0]
        && name == "Rational"
        && rat_args.len() == 2
        && let (Expr::Integer(n), Expr::Integer(d)) =
          (&rat_args[0], &rat_args[1])
      {
        // Compute triangle wave for n/d exactly
        // shifted = n/d + 3/4 = (4n + 3d) / (4d)
        let num = 4 * n + 3 * d;
        let den = 4 * d;
        // frac = num mod den / den (using Euclidean remainder)
        let rem = num.rem_euclid(den);
        // val = 4 * |rem/den - 1/2| - 1 = 4 * |rem - den/2| / den - 1
        // = (4 * |2*rem - den|) / (2*den) - 1
        // = (2 * |2*rem - den| - den) / den
        let two_rem_minus_den = 2 * rem - den;
        let abs_val = two_rem_minus_den.abs();
        let result_num = 2 * abs_val - den;
        let result_den = den;
        // Simplify result_num / result_den
        let g = gcd(result_num, result_den);
        let sn = result_num / g;
        let sd = result_den / g;
        if sd == 1 {
          return Ok(Expr::Integer(sn));
        }
        return Ok(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(sn), Expr::Integer(sd)],
        });
      }
      // Float input
      if let Some(t) = expr_to_f64(&args[0]) {
        let shifted = t + 0.75;
        let frac = shifted - shifted.floor();
        let val = 4.0 * (frac - 0.5).abs() - 1.0;
        return Ok(Expr::Real(val));
      }
      Ok(Expr::FunctionCall {
        name: "TriangleWave".to_string(),
        args: args.to_vec(),
      })
    }
    2 => {
      // TriangleWave[{min, max}, t]
      if let Expr::List(bounds) = &args[0]
        && bounds.len() == 2
      {
        // First compute base triangle wave value
        let base_args = [args[1].clone()];
        let base = triangle_wave_ast(&base_args)?;
        if let Some(v) = expr_to_f64(&base)
          && let (Some(lo), Some(hi)) =
            (expr_to_f64(&bounds[0]), expr_to_f64(&bounds[1]))
        {
          // Scale from [-1,1] to [min,max]: result = min + (max-min)*(v+1)/2
          let result = lo + (hi - lo) * (v + 1.0) / 2.0;
          return Ok(Expr::Real(result));
        }
      }
      Ok(Expr::FunctionCall {
        name: "TriangleWave".to_string(),
        args: args.to_vec(),
      })
    }
    _ => Err(InterpreterError::EvaluationError(
      "TriangleWave expects 1 or 2 arguments".into(),
    )),
  }
}

/// SawtoothWave[x] - periodic sawtooth wave, fractional part of x
pub fn sawtooth_wave_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match args.len() {
    1 => {
      // Integer input: SawtoothWave[n] = 0 for all integers
      if let Expr::Integer(_) = &args[0] {
        return Ok(Expr::Integer(0));
      }
      // Exact rational input: fractional part of n/d
      if let Expr::FunctionCall {
        name,
        args: rat_args,
      } = &args[0]
        && name == "Rational"
        && rat_args.len() == 2
        && let (Expr::Integer(n), Expr::Integer(d)) =
          (&rat_args[0], &rat_args[1])
      {
        // frac(n/d) = (n mod d) / d using Euclidean remainder
        let rem = n.rem_euclid(*d);
        if rem == 0 {
          return Ok(Expr::Integer(0));
        }
        let g = gcd(rem, *d);
        let sn = rem / g;
        let sd = d / g;
        if sd == 1 {
          return Ok(Expr::Integer(sn));
        }
        return Ok(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(sn), Expr::Integer(sd)],
        });
      }
      // Float input
      if let Some(t) = expr_to_f64(&args[0]) {
        let frac = t - t.floor();
        return Ok(Expr::Real(frac));
      }
      Ok(Expr::FunctionCall {
        name: "SawtoothWave".to_string(),
        args: args.to_vec(),
      })
    }
    2 => {
      // SawtoothWave[{min, max}, t]
      if let Expr::List(bounds) = &args[0]
        && bounds.len() == 2
      {
        let base_args = [args[1].clone()];
        let base = sawtooth_wave_ast(&base_args)?;
        if let Some(v) = expr_to_f64(&base)
          && let (Some(lo), Some(hi)) =
            (expr_to_f64(&bounds[0]), expr_to_f64(&bounds[1]))
        {
          // Scale from [0,1] to [min,max]: result = min + (max-min)*v
          let result = lo + (hi - lo) * v;
          return Ok(Expr::Real(result));
        }
      }
      Ok(Expr::FunctionCall {
        name: "SawtoothWave".to_string(),
        args: args.to_vec(),
      })
    }
    _ => Err(InterpreterError::EvaluationError(
      "SawtoothWave expects 1 or 2 arguments".into(),
    )),
  }
}

/// ParabolicCylinderD[ν, z] - parabolic cylinder function D_ν(z)
pub fn parabolic_cylinder_d_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ParabolicCylinderD expects exactly 2 arguments".into(),
    ));
  }

  // Numeric evaluation when both arguments are numeric
  if let (Some(nu), Some(z)) = (expr_to_f64(&args[0]), expr_to_f64(&args[1]))
    && (matches!(&args[0], Expr::Real(_)) || matches!(&args[1], Expr::Real(_)))
  {
    return Ok(Expr::Real(parabolic_cylinder_d(nu, z)));
  }

  Ok(Expr::FunctionCall {
    name: "ParabolicCylinderD".to_string(),
    args: args.to_vec(),
  })
}

/// Compute D_ν(z) using the relation to confluent hypergeometric functions:
/// D_ν(z) = 2^(ν/2) * exp(-z²/4) * [
///   √π / Γ((1-ν)/2) * 1F1(-ν/2, 1/2, z²/2)
///   - √2 * z / Γ(-ν/2) * 1F1((1-ν)/2, 3/2, z²/2)
/// ]
pub fn parabolic_cylinder_d(nu: f64, z: f64) -> f64 {
  use std::f64::consts::PI;
  let z2_half = z * z / 2.0;
  let prefactor = 2.0_f64.powf(nu / 2.0) * PI.sqrt() * (-z * z / 4.0).exp();

  let gamma_1 = gamma_fn((1.0 - nu) / 2.0);
  let gamma_2 = gamma_fn(-nu / 2.0);

  let term1 = if gamma_1.is_finite() && gamma_1.abs() > 1e-300 {
    hypergeometric_1f1(-nu / 2.0, 0.5, z2_half) / gamma_1
  } else {
    0.0
  };

  let term2 = if gamma_2.is_finite() && gamma_2.abs() > 1e-300 {
    -2.0_f64.sqrt() * z / gamma_2
      * hypergeometric_1f1((1.0 - nu) / 2.0, 1.5, z2_half)
  } else {
    0.0
  };

  prefactor * (term1 + term2)
}

/// AngerJ[nu, z] — Anger function.
///
/// For integer nu, AngerJ[n, z] = BesselJ[n, z].
/// For general nu, AngerJ[nu, z] = (1/Pi) * Integral[Cos[nu*t - z*Sin[t]], {t, 0, Pi}]
pub fn anger_j_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "AngerJ expects exactly 2 arguments".into(),
    ));
  }
  let nu_expr = &args[0];
  let z_expr = &args[1];

  // Extract numeric values
  let nu_val = match nu_expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  };
  let z_val = match z_expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  };

  // Special case: AngerJ[n, 0] for integer n
  if matches!(z_expr, Expr::Integer(0))
    && let Expr::Integer(n) = nu_expr
  {
    return if *n == 0 {
      Ok(Expr::Integer(1))
    } else {
      Ok(Expr::Integer(0))
    };
  }

  // For integer nu, AngerJ[n, z] = BesselJ[n, z]
  if let Expr::Integer(_) = nu_expr {
    return bessel_j_ast(args);
  }

  // Numeric evaluation when both args are numeric and at least one is Real
  let is_numeric_eval = nu_val.is_some()
    && z_val.is_some()
    && (matches!(z_expr, Expr::Real(_)) || matches!(nu_expr, Expr::Real(_)));

  if is_numeric_eval {
    let nu = nu_val.unwrap();
    let z = z_val.unwrap();
    let result = anger_j(nu, z);
    return Ok(Expr::Real(result));
  }

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "AngerJ".to_string(),
    args: args.to_vec(),
  })
}

/// Compute AngerJ[nu, z] numerically using Gauss-Legendre quadrature.
///
/// AngerJ[nu, z] = (1/Pi) * Integral[Cos[nu*t - z*Sin[t]], {t, 0, Pi}]
/// For integer nu, this equals BesselJ[n, z].
pub fn anger_j(nu: f64, z: f64) -> f64 {
  // For integer nu, delegate to BesselJ
  if nu == nu.floor() && nu.is_finite() {
    return bessel_j(nu, z);
  }

  // For z = 0: AngerJ[nu, 0] = Sin[nu*Pi] / (nu*Pi)
  if z == 0.0 {
    let x = nu * std::f64::consts::PI;
    return x.sin() / x;
  }

  // Gauss-Legendre quadrature on [0, Pi]
  // Transform: t = (Pi/2) * (u + 1) where u in [-1, 1]
  gauss_legendre_anger_j(nu, z)
}

/// Gauss-Legendre quadrature for the Anger function integral.
fn gauss_legendre_anger_j(nu: f64, z: f64) -> f64 {
  // Use 64-point Gauss-Legendre quadrature
  // Nodes and weights for [-1, 1]
  let nodes_weights = gauss_legendre_64();
  let half_pi = std::f64::consts::PI / 2.0;
  let inv_pi = 1.0 / std::f64::consts::PI;

  let mut sum = 0.0;
  for &(node, weight) in &nodes_weights {
    let t = half_pi * (node + 1.0); // Map [-1,1] to [0, Pi]
    let integrand = (nu * t - z * t.sin()).cos();
    sum += weight * integrand;
  }
  sum * half_pi * inv_pi
}

/// WeberE[nu, z] — Weber function.
///
/// E_nu(z) = (1/Pi) * Integral[Sin[nu*t - z*Sin[t]], {t, 0, Pi}]
pub fn weber_e_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "WeberE expects exactly 2 arguments".into(),
    ));
  }
  let nu_expr = &args[0];
  let z_expr = &args[1];

  let nu_val = match nu_expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  };
  let z_val = match z_expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  };

  // Special case: WeberE[0, 0] = 0
  if matches!(z_expr, Expr::Integer(0)) && matches!(nu_expr, Expr::Integer(0)) {
    return Ok(Expr::Integer(0));
  }

  // Numeric evaluation when both args are numeric and at least one is Real
  let is_numeric_eval = nu_val.is_some()
    && z_val.is_some()
    && (matches!(z_expr, Expr::Real(_)) || matches!(nu_expr, Expr::Real(_)));

  if is_numeric_eval {
    let nu = nu_val.unwrap();
    let z = z_val.unwrap();
    let result = weber_e(nu, z);
    return Ok(Expr::Real(result));
  }

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "WeberE".to_string(),
    args: args.to_vec(),
  })
}

/// Compute WeberE[nu, z] numerically using Gauss-Legendre quadrature.
///
/// E_nu(z) = (1/Pi) * Integral[Sin[nu*t - z*Sin[t]], {t, 0, Pi}]
pub fn weber_e(nu: f64, z: f64) -> f64 {
  // For z = 0: WeberE[nu, 0] = (1 - Cos[nu*Pi]) / (nu*Pi)
  if z == 0.0 {
    if nu == 0.0 {
      return 0.0;
    }
    let x = nu * std::f64::consts::PI;
    return (1.0 - x.cos()) / x;
  }

  // Gauss-Legendre quadrature
  let nodes_weights = gauss_legendre_64();
  let half_pi = std::f64::consts::PI / 2.0;
  let inv_pi = 1.0 / std::f64::consts::PI;

  let mut sum = 0.0;
  for &(node, weight) in &nodes_weights {
    let t = half_pi * (node + 1.0);
    let integrand = (nu * t - z * t.sin()).sin();
    sum += weight * integrand;
  }
  sum * half_pi * inv_pi
}

/// WignerD[{j, m1, m2}, theta] — Wigner d-matrix element d^j_{m1,m2}(theta).
///
/// WignerD[{j, m1, m2}, phi, theta, psi] — full Wigner D-matrix element:
///   D^j_{m1,m2}(phi,theta,psi) = E^(-I*m1*phi) * d^j_{m1,m2}(theta) * E^(-I*m2*psi)
pub fn wigner_d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Supported forms:
  // WignerD[{j, m1, m2}, theta]
  // WignerD[{j, m1, m2}, phi, theta, psi]
  if args.len() != 2 && args.len() != 4 {
    return Ok(Expr::FunctionCall {
      name: "WignerD".to_string(),
      args: args.to_vec(),
    });
  }

  let (j_val, m1_val, m2_val) = match &args[0] {
    Expr::List(items) if items.len() == 3 => {
      let j = match try_eval_to_f64(&items[0]) {
        Some(v) => v,
        None => {
          return Ok(Expr::FunctionCall {
            name: "WignerD".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let m1 = match try_eval_to_f64(&items[1]) {
        Some(v) => v,
        None => {
          return Ok(Expr::FunctionCall {
            name: "WignerD".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let m2 = match try_eval_to_f64(&items[2]) {
        Some(v) => v,
        None => {
          return Ok(Expr::FunctionCall {
            name: "WignerD".to_string(),
            args: args.to_vec(),
          });
        }
      };
      (j, m1, m2)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "WignerD".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if args.len() == 2 {
    // WignerD[{j, m1, m2}, theta]
    let theta = match try_eval_to_f64(&args[1]) {
      Some(v) => v,
      None => {
        // Check if at least one is Real
        if !matches!(&args[1], Expr::Real(_)) {
          return Ok(Expr::FunctionCall {
            name: "WignerD".to_string(),
            args: args.to_vec(),
          });
        }
        return Ok(Expr::FunctionCall {
          name: "WignerD".to_string(),
          args: args.to_vec(),
        });
      }
    };
    let result = wigner_d_small(j_val, m1_val, m2_val, theta);
    Ok(Expr::Real(result))
  } else {
    // WignerD[{j, m1, m2}, phi, theta, psi]
    let phi = match try_eval_to_f64(&args[1]) {
      Some(v) => v,
      None => {
        return Ok(Expr::FunctionCall {
          name: "WignerD".to_string(),
          args: args.to_vec(),
        });
      }
    };
    let theta = match try_eval_to_f64(&args[2]) {
      Some(v) => v,
      None => {
        return Ok(Expr::FunctionCall {
          name: "WignerD".to_string(),
          args: args.to_vec(),
        });
      }
    };
    let psi = match try_eval_to_f64(&args[3]) {
      Some(v) => v,
      None => {
        return Ok(Expr::FunctionCall {
          name: "WignerD".to_string(),
          args: args.to_vec(),
        });
      }
    };
    let d = wigner_d_small(j_val, m1_val, m2_val, theta);
    // D = E^(-I*m1*phi) * d * E^(-I*m2*psi)
    // For real angles, this is: d * E^(-I*(m1*phi + m2*psi))
    // = d * (cos(m1*phi+m2*psi) - I*sin(m1*phi+m2*psi))
    let phase = m1_val * phi + m2_val * psi;
    let re = d * phase.cos();
    let im = -d * phase.sin();
    if im.abs() < 1e-15 {
      Ok(Expr::Real(re))
    } else {
      // Return Complex form
      Ok(Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          Expr::Real(re),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Real(im), Expr::Identifier("I".to_string())],
          },
        ],
      })
    }
  }
}

/// Compute the Wigner (small) d-matrix element d^j_{m1,m2}(theta).
fn wigner_d_small(j: f64, m1: f64, m2: f64, theta: f64) -> f64 {
  // Only handle half-integer/integer j, m1, m2
  let j2 = (2.0 * j).round() as i64;
  let m1_2 = (2.0 * m1).round() as i64;
  let m2_2 = (2.0 * m2).round() as i64;

  // Validate
  if (j2 as f64 - 2.0 * j).abs() > 1e-10
    || (m1_2 as f64 - 2.0 * m1).abs() > 1e-10
    || (m2_2 as f64 - 2.0 * m2).abs() > 1e-10
  {
    return f64::NAN;
  }

  if m1_2.abs() > j2 || m2_2.abs() > j2 {
    return 0.0;
  }

  // Use integer arithmetic for factorials
  // s ranges from max(0, m1-m2) to min(j+m1, j-m2)
  // Using half-integer labels: j+m1 = (j2+m1_2)/2, etc.
  let jpm1 = (j2 + m1_2) / 2;
  let jmm1 = (j2 - m1_2) / 2;
  let jpm2 = (j2 + m2_2) / 2;
  let jmm2 = (j2 - m2_2) / 2;
  let m1mm2 = (m1_2 - m2_2) / 2;

  let s_min = 0i64.max(m1mm2);
  let s_max = jpm1.min(jmm2);

  if s_min > s_max {
    return 0.0;
  }

  let half_theta = theta / 2.0;
  let cos_ht = half_theta.cos();
  let sin_ht = half_theta.sin();

  let prefactor = (factorial_f64(jpm1 as u64)
    * factorial_f64(jmm1 as u64)
    * factorial_f64(jpm2 as u64)
    * factorial_f64(jmm2 as u64))
  .sqrt();

  let mut sum = 0.0;
  for s in s_min..=s_max {
    let denom = factorial_f64((jpm1 - s) as u64)
      * factorial_f64(s as u64)
      * factorial_f64((s - m1mm2) as u64)
      * factorial_f64((jmm2 - s) as u64);

    // cos(theta/2)^(2j + m1 - m2 - 2s) * sin(theta/2)^(m2 - m1 + 2s)
    let cos_power = (2.0 * j + m1 - m2 - 2.0 * s as f64) as i64;
    let sin_power = (m2 - m1 + 2.0 * s as f64) as i64;

    let sign = if s % 2 == 0 { 1.0 } else { -1.0 };
    let term =
      sign * cos_ht.powi(cos_power as i32) * sin_ht.powi(sin_power as i32)
        / denom;
    sum += term;
  }

  prefactor * sum
}

/// Factorial as f64 for moderate n.
fn factorial_f64(n: u64) -> f64 {
  if n <= 1 {
    return 1.0;
  }
  let mut result = 1.0;
  for i in 2..=n {
    result *= i as f64;
  }
  result
}

/// 64-point Gauss-Legendre nodes and weights on [-1, 1].
fn gauss_legendre_64() -> Vec<(f64, f64)> {
  // Compute nodes and weights using the Golub-Welsch algorithm
  let n = 64;
  let mut nodes = vec![0.0_f64; n];
  let mut weights = vec![0.0_f64; n];

  for i in 0..n {
    // Initial guess using Chebyshev nodes
    let mut x =
      -(std::f64::consts::PI * (4 * i + 3) as f64 / (4 * n + 2) as f64).cos();

    // Newton's method to find roots of P_n(x)
    for _ in 0..100 {
      let (p, dp) = legendre_p_and_deriv(n, x);
      let dx = -p / dp;
      x += dx;
      if dx.abs() < 1e-16 {
        break;
      }
    }
    nodes[i] = x;
    let (_, dp) = legendre_p_and_deriv(n, x);
    weights[i] = 2.0 / ((1.0 - x * x) * dp * dp);
  }

  nodes.into_iter().zip(weights).collect()
}

/// Evaluate Legendre polynomial P_n(x) and its derivative P_n'(x).
fn legendre_p_and_deriv(n: usize, x: f64) -> (f64, f64) {
  let mut p0 = 1.0;
  let mut p1 = x;
  let mut dp0 = 0.0;
  let mut dp1 = 1.0;

  for k in 1..n {
    let kf = k as f64;
    let p2 = ((2.0 * kf + 1.0) * x * p1 - kf * p0) / (kf + 1.0);
    let dp2 = ((2.0 * kf + 1.0) * (p1 + x * dp1) - kf * dp0) / (kf + 1.0);
    p0 = p1;
    p1 = p2;
    dp0 = dp1;
    dp1 = dp2;
  }

  (p1, dp1)
}

/// NorlundB[n, a] - Nörlund generalized Bernoulli polynomial B_n^(a).
/// Computed via power series: (t/(e^t-1))^a = sum h_k t^k, B_n^(a) = n! * h_n.
/// Each h_k is a polynomial in a with rational coefficients.
pub fn norlund_b_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "NorlundB".to_string(),
      args: args.to_vec(),
    });
  }

  let n = match &args[0] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "NorlundB".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let a_expr = &args[1];

  // Compute B_n^(a) as polynomial in a: vec of (num, den) pairs for a^0, a^1, ..., a^n
  let poly = norlund_b_poly(n);

  // If a is numeric, evaluate the polynomial
  if let Some(a_val) = try_eval_to_f64(a_expr) {
    // Check if a is an integer for exact computation
    if a_val == a_val.round() && a_val.abs() < 1e15 {
      let a_int = a_val as i128;
      // Evaluate polynomial at integer a using exact arithmetic
      let mut result_n: i128 = 0;
      let mut result_d: i128 = 1;
      let mut a_pow: i128 = 1; // a^k
      for &(cn, cd) in &poly {
        if cn != 0 {
          // result += cn/cd * a^k
          let term_n = cn.checked_mul(a_pow);
          if let Some(tn) = term_n {
            let new_n = result_n
              .checked_mul(cd)
              .and_then(|x| x.checked_add(tn.checked_mul(result_d)?));
            let new_d = result_d.checked_mul(cd);
            if let (Some(nn), Some(nd)) = (new_n, new_d) {
              let g = gcd(nn.abs(), nd.abs());
              result_n = nn / g;
              result_d = nd / g;
            } else {
              // Overflow - fall through to symbolic
              return evaluate_norlund_symbolic(&poly, a_expr);
            }
          } else {
            return evaluate_norlund_symbolic(&poly, a_expr);
          }
        }
        a_pow = match a_pow.checked_mul(a_int) {
          Some(v) => v,
          None => return evaluate_norlund_symbolic(&poly, a_expr),
        };
      }
      if result_d < 0 {
        result_n = -result_n;
        result_d = -result_d;
      }
      return Ok(crate::functions::math_ast::make_rational(
        result_n, result_d,
      ));
    }
  }

  // Symbolic case: build the polynomial expression
  evaluate_norlund_symbolic(&poly, a_expr)
}

/// Build the symbolic polynomial expression from coefficients.
fn evaluate_norlund_symbolic(
  poly: &[(i128, i128)],
  a: &Expr,
) -> Result<Expr, InterpreterError> {
  let mut terms: Vec<Expr> = Vec::new();
  for (k, &(cn, cd)) in poly.iter().enumerate() {
    if cn == 0 {
      continue;
    }
    let coeff = crate::functions::math_ast::make_rational(cn, cd);
    let term = if k == 0 {
      coeff
    } else {
      let a_pow = if k == 1 {
        a.clone()
      } else {
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![a.clone(), Expr::Integer(k as i128)],
        }
      };
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![coeff, a_pow],
      }
    };
    terms.push(term);
  }
  if terms.is_empty() {
    return Ok(Expr::Integer(0));
  }
  if terms.len() == 1 {
    return crate::evaluator::evaluate_expr_to_expr(&terms.pop().unwrap());
  }
  let sum = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms,
  };
  crate::evaluator::evaluate_expr_to_expr(&sum)
}

/// Compute B_n^(a) as polynomial in a: returns vec of (num, den) for coefficients of a^0, a^1, ..., a^n.
/// Uses power series: f(t) = t/(e^t-1), h_k = coeff of t^k in f(t)^a, B_n^(a) = n! * h_n.
/// Recurrence: h_0 = 1, h_k = (1/k) * sum_{j=1}^{k} ((a+1)*j - k) * s_j * h_{k-j}
/// where s_j = B_j / j! (Bernoulli number divided by factorial).
fn norlund_b_poly(n: usize) -> Vec<(i128, i128)> {
  // s_j = B_j / j! as (numerator, denominator)
  let mut s: Vec<(i128, i128)> = Vec::with_capacity(n + 1);
  let mut factorial: i128 = 1;
  for j in 0..=n {
    if j > 0 {
      factorial *= j as i128;
    }
    if let Some((bn, bd)) = bernoulli_number(j) {
      let g = gcd(bn.abs(), factorial.abs());
      s.push((bn / g, bd * (factorial / g)));
    } else {
      s.push((0, 1));
    }
  }
  // Simplify s values
  for sval in &mut s {
    let g = gcd(sval.0.abs(), sval.1.abs());
    if g > 0 {
      sval.0 /= g;
      sval.1 /= g;
    }
    if sval.1 < 0 {
      sval.0 = -sval.0;
      sval.1 = -sval.1;
    }
  }

  // h_k is a polynomial in a of degree k, stored as Vec<(i128, i128)>
  // h_k[i] = coefficient of a^i
  let mut h: Vec<Vec<(i128, i128)>> = Vec::with_capacity(n + 1);
  h.push(vec![(1, 1)]); // h_0 = 1

  for k in 1..=n {
    // h_k = (1/k) * sum_{j=1}^{k} ((a+1)*j - k) * s_j * h_{k-j}
    //      = (1/k) * sum_{j=1}^{k} s_j * (j*a*h_{k-j} + (j-k)*h_{k-j})
    let max_deg = k;
    let mut hk = vec![(0i128, 1i128); max_deg + 1];

    for j in 1..=k {
      let (sn, sd) = s[j];
      if sn == 0 {
        continue;
      }
      let h_prev = &h[k - j];

      // Add s_j * (j-k) * h_{k-j} to hk (no shift in a)
      let scale = (j as i128) - (k as i128); // j - k
      if scale != 0 {
        for (i, &(pn, pd)) in h_prev.iter().enumerate() {
          if pn == 0 {
            continue;
          }
          // term = sn/sd * scale * pn/pd = (sn * scale * pn) / (sd * pd)
          let tn = sn * scale * pn;
          let td = sd * pd;
          rat_add_inplace(&mut hk[i], tn, td);
        }
      }

      // Add s_j * j * a * h_{k-j} to hk (shift by 1 in a)
      let j_val = j as i128;
      for (i, &(pn, pd)) in h_prev.iter().enumerate() {
        if pn == 0 {
          continue;
        }
        let tn = sn * j_val * pn;
        let td = sd * pd;
        rat_add_inplace(&mut hk[i + 1], tn, td);
      }
    }

    // Divide by k
    for coeff in &mut hk {
      coeff.1 *= k as i128;
      let g = gcd(coeff.0.abs(), coeff.1.abs());
      if g > 0 {
        coeff.0 /= g;
        coeff.1 /= g;
      }
      if coeff.1 < 0 {
        coeff.0 = -coeff.0;
        coeff.1 = -coeff.1;
      }
    }

    h.push(hk);
  }

  // B_n^(a) = n! * h_n
  let mut factorial: i128 = 1;
  for i in 1..=n {
    factorial *= i as i128;
  }
  let mut result = h[n].clone();
  for coeff in &mut result {
    coeff.0 *= factorial;
    let g = gcd(coeff.0.abs(), coeff.1.abs());
    if g > 0 {
      coeff.0 /= g;
      coeff.1 /= g;
    }
    if coeff.1 < 0 {
      coeff.0 = -coeff.0;
      coeff.1 = -coeff.1;
    }
  }
  result
}

/// Add rational tn/td to the value at *target in-place.
fn rat_add_inplace(target: &mut (i128, i128), tn: i128, td: i128) {
  let (rn, rd) = target;
  let new_n = *rn * td + tn * *rd;
  let new_d = *rd * td;
  let g = gcd(new_n.abs(), new_d.abs());
  if g > 0 {
    *rn = new_n / g;
    *rd = new_d / g;
  } else {
    *rn = new_n;
    *rd = new_d;
  }
  if *rd < 0 {
    *rn = -*rn;
    *rd = -*rd;
  }
}
