#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;

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

  // If any upper parameter is exactly 0, (0)_n = 0 for n >= 1, so the series
  // collapses to its n = 0 term which is 1.
  if a_list.iter().any(|a| matches!(a, Expr::Integer(0))) {
    return Ok(Expr::Integer(1));
  }

  // HypergeometricPFQ[{}, {}, z] = E^z
  if a_list.is_empty() && b_list.is_empty() {
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![Expr::Identifier("E".to_string()), z.clone()],
    });
  }

  // When each upper parameter cancels a lower parameter (identical
  // multisets), (a)_n / (a)_n = 1 and the series reduces to Σ z^n/n! = E^z.
  if a_list.len() == b_list.len() && !a_list.is_empty() {
    let key = |e: &Expr| crate::syntax::expr_to_string(e);
    let mut a_keys: Vec<String> = a_list.iter().map(key).collect();
    let mut b_keys: Vec<String> = b_list.iter().map(key).collect();
    a_keys.sort();
    b_keys.sort();
    if a_keys == b_keys {
      return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![Expr::Identifier("E".to_string()), z.clone()],
      });
    }
  }

  // Closed form: 1F1[b+1, b, z] = ((b + z) / b) * E^z for positive integer b.
  if a_list.len() == 1
    && b_list.len() == 1
    && let (Expr::Integer(a_i), Expr::Integer(b_i)) = (&a_list[0], &b_list[0])
    && *b_i >= 1
    && *a_i == *b_i + 1
  {
    let b_int = *b_i;
    let plus = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![Expr::Integer(b_int), z.clone()],
    };
    let ratio = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(plus),
      right: Box::new(Expr::Integer(b_int)),
    };
    let exp_z = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![Expr::Identifier("E".to_string()), z.clone()],
    };
    let prod = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(ratio),
      right: Box::new(exp_z),
    };
    return crate::evaluator::evaluate_expr_to_expr(&prod);
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
    // Convergence at the boundary |z|=1 is a concern only for p = q+1
    // (the 2F1-like balanced case): the series converges there iff
    // Re(Σb − Σa) > 0. For p ≤ q the series has infinite convergence radius,
    // so values at |z|=1 are finite (e.g. HypergeometricPFQ[{3},{2},1] = 3e/2).
    if z_val.abs() == 1.0 && a_vals.len() == b_vals.len() + 1 {
      let sum_a: f64 = a_vals.iter().sum();
      let sum_b: f64 = b_vals.iter().sum();
      if sum_b - sum_a <= 0.0 {
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

/// Returns Binomial[n, k] / k! as a reduced (numerator, denominator) pair.
fn binomial_over_factorial(n: u32, k: u32) -> (i128, i128) {
  // Binomial[n, k] = n! / (k! * (n-k)!), so Binomial[n, k] / k! has
  // numerator n! / (n-k)! and denominator k!·k!.
  let mut num: i128 = 1;
  for i in 0..k {
    num *= (n - i) as i128;
  }
  let mut den: i128 = 1;
  for i in 1..=k {
    den *= i as i128;
  }
  // Also divide by k! again (k!·k! total)
  let kfact: i128 = (1..=k).map(|i| i as i128).product::<i128>().max(1);
  den *= kfact;
  let g = gcd_i128(num.abs(), den.abs());
  (num / g, den / g)
}

fn gcd_i128(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a, b);
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a.max(1)
}

/// Hypergeometric1F1[a, b, z] - Kummer's confluent hypergeometric function
fn is_half(expr: &Expr) -> bool {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      matches!(&args[0], Expr::Integer(1))
        && matches!(&args[1], Expr::Integer(2))
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left,
      right,
    } => {
      matches!(left.as_ref(), Expr::Integer(1))
        && matches!(right.as_ref(), Expr::Integer(2))
    }
    _ => false,
  }
}

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

  // 1F1[0, 0, z] = 1 — the Pochhammer ratio (0)_k/(0)_k is conventionally 0
  // for k ≥ 1, so only the k=0 term survives. (Distinct from the general
  // 1F1[a, a, z] = E^z rule; here a = 0 collapses to the constant 1.)
  if is_expr_zero(&args[0]) && is_expr_zero(&args[1]) {
    if matches!(&args[2], Expr::Real(_) | Expr::BigFloat(_, _)) {
      return Ok(Expr::Real(1.0));
    }
    return Ok(Expr::Integer(1));
  }

  // 1F1[a, a, z] = E^z (the (a)_n factors cancel). Use Exp to trigger the
  // numeric-path evaluation when z is real and threading over lists.
  if crate::syntax::expr_to_string(&args[0])
    == crate::syntax::expr_to_string(&args[1])
  {
    let exp_call = Expr::FunctionCall {
      name: "Exp".to_string(),
      args: vec![args[2].clone()],
    };
    return crate::evaluator::evaluate_expr_to_expr(&exp_call);
  }

  // Kummer identity: 1F1[1/2, 1, z] = E^(z/2) * BesselI[0, z/2].
  if is_half(&args[0]) && matches!(&args[1], Expr::Integer(1)) {
    let z = &args[2];
    let half_z = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(z.clone()),
      right: Box::new(Expr::Integer(2)),
    };
    let exp_part = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: Box::new(Expr::Constant("E".to_string())),
      right: Box::new(half_z.clone()),
    };
    let bessel = Expr::FunctionCall {
      name: "BesselI".to_string(),
      args: vec![Expr::Integer(0), half_z],
    };
    return crate::evaluator::evaluate_function_call_ast(
      "Times",
      &[exp_part, bessel],
    );
  }

  // Identity: 1F1[1, 1/2, z] = 1 + E^z · √π · √z · Erf[√z].
  // Skip for machine-precision numeric z; the series/numeric path below
  // produces a plain Real there, matching wolframscript.
  if matches!(&args[0], Expr::Integer(1))
    && is_half(&args[1])
    && !matches!(&args[2], Expr::Real(_) | Expr::BigFloat(_, _))
  {
    let z = &args[2];
    let exp_z = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: Box::new(Expr::Constant("E".to_string())),
      right: Box::new(z.clone()),
    };
    let sqrt_pi = Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: vec![Expr::Constant("Pi".to_string())],
    };
    let sqrt_z = Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: vec![z.clone()],
    };
    let erf = Expr::FunctionCall {
      name: "Erf".to_string(),
      args: vec![sqrt_z.clone()],
    };
    let product = crate::evaluator::evaluate_function_call_ast(
      "Times",
      &[exp_z, sqrt_pi, sqrt_z, erf],
    )?;
    return crate::evaluator::evaluate_function_call_ast(
      "Plus",
      &[Expr::Integer(1), product],
    );
  }

  // Closed form for 1F1[positive integer a, 1, z]:
  //   E^z * Σ_{k=0}^{a-1} Binomial[a-1, k] z^k / k!
  // (Kummer's transformation plus the Laguerre identity 1F1[-n, 1, z] = L_n(z)
  // — folding the Laguerre polynomial back into its explicit form.)
  if let (Expr::Integer(a), Expr::Integer(b)) = (&args[0], &args[1])
    && *a >= 1
    && *b == 1
    && *a <= 20
  {
    let n = (*a - 1) as u32;
    let z = &args[2];
    let mut terms: Vec<Expr> = Vec::with_capacity((n + 1) as usize);
    for k in 0..=n {
      let coeff = binomial_over_factorial(n, k);
      let z_pow = if k == 0 {
        Expr::Integer(1)
      } else if k == 1 {
        z.clone()
      } else {
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![z.clone(), Expr::Integer(k as i128)],
        }
      };
      let term = match coeff {
        (1, 1) => z_pow,
        (num, 1) => crate::evaluator::evaluate_function_call_ast(
          "Times",
          &[Expr::Integer(num), z_pow],
        )?,
        (num, den) => {
          let rat = crate::functions::math_ast::make_rational_pub(num, den);
          crate::evaluator::evaluate_function_call_ast("Times", &[rat, z_pow])?
        }
      };
      terms.push(term);
    }
    let polynomial =
      crate::evaluator::evaluate_function_call_ast("Plus", &terms)?;
    let exp_part =
      crate::evaluator::evaluate_function_call_ast("Exp", &[z.clone()])?;
    return crate::evaluator::evaluate_function_call_ast(
      "Times",
      &[exp_part, polynomial],
    );
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

  // Special case: HypergeometricU[0, b, z] = 1
  if matches!(&args[0], Expr::Integer(0)) {
    return Ok(Expr::Integer(1));
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

/// Hypergeometric1F1Regularized[a, b, z] = Hypergeometric1F1[a, b, z] / Gamma[b]
pub fn hypergeometric_1f1_regularized_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "Hypergeometric1F1Regularized expects exactly 3 arguments".into(),
    ));
  }

  // Numeric evaluation when all args are numeric and at least one is Real
  let a_val = expr_to_f64(&args[0]);
  let b_val = expr_to_f64(&args[1]);
  let z_val = expr_to_f64(&args[2]);
  let has_real = matches!(args[0], Expr::Real(_))
    || matches!(args[1], Expr::Real(_))
    || matches!(args[2], Expr::Real(_));

  if let (Some(_a), Some(b), Some(_z)) = (a_val, b_val, z_val)
    && has_real
  {
    // Compute 1F1(a, b, z) then divide by Gamma(b)
    let h1f1 = crate::evaluator::evaluate_function_call_ast(
      "Hypergeometric1F1",
      &[
        Expr::Real(a_val.unwrap()),
        Expr::Real(b),
        Expr::Real(z_val.unwrap()),
      ],
    )?;
    if let Some(h_val) = expr_to_f64(&h1f1) {
      let gamma_b = lgamma(b).exp();
      return Ok(Expr::Real(h_val / gamma_b));
    }
  }

  // Unevaluated
  Ok(Expr::FunctionCall {
    name: "Hypergeometric1F1Regularized".to_string(),
    args: args.to_vec(),
  })
}
