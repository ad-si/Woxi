#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, expr_to_string};

/// PolynomialRemainder[p, q, x] - remainder of polynomial division
pub fn polynomial_remainder_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "PolynomialRemainder expects 3 arguments".into(),
    ));
  }
  let var = match &args[2] {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PolynomialRemainder".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let (_, remainder) = poly_divide_symbolic(&args[0], &args[1], var)?;
  crate::evaluator::evaluate_expr_to_expr(&remainder)
}

/// PolynomialQuotient[p, q, x] - quotient of polynomial division
pub fn polynomial_quotient_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "PolynomialQuotient expects 3 arguments".into(),
    ));
  }
  let var = match &args[2] {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PolynomialQuotient".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let (quotient, _) = poly_divide_symbolic(&args[0], &args[1], var)?;
  crate::evaluator::evaluate_expr_to_expr(&quotient)
}

/// PolynomialQuotientRemainder[p, q, x] - {quotient, remainder} of polynomial division
pub fn polynomial_quotient_remainder_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "PolynomialQuotientRemainder expects 3 arguments".into(),
    ));
  }
  let var = match &args[2] {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PolynomialQuotientRemainder".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let (quotient, remainder) = poly_divide_symbolic(&args[0], &args[1], var)?;
  let q = crate::evaluator::evaluate_expr_to_expr(&quotient)?;
  let r = crate::evaluator::evaluate_expr_to_expr(&remainder)?;
  Ok(Expr::List(vec![q, r].into()))
}

/// PolynomialReduce[poly, {p1, p2, ...}, x] — reduce `poly` modulo the
/// polynomials `pi` in the single variable `x`. Returns `{{a1, a2, ...}, b}`
/// where `a1 p1 + a2 p2 + ... + b == poly` and `b` is the (minimal) remainder.
///
/// Only the single-variable case is handled; a multivariable variable list is
/// returned unevaluated.
pub fn polynomial_reduce_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "PolynomialReduce".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() != 3 {
    return unevaluated();
  }
  let divisors: Vec<Expr> = match &args[1] {
    Expr::List(items) => items.to_vec(),
    _ => return unevaluated(),
  };
  let var = match &args[2] {
    Expr::Identifier(v) => v.clone(),
    Expr::List(vs) if vs.len() == 1 => match &vs[0] {
      Expr::Identifier(v) => v.clone(),
      _ => return unevaluated(),
    },
    _ => return unevaluated(),
  };

  let eval = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e);
  let var_id = Expr::Identifier(var.clone());
  let expand = |e: &Expr| -> Expr {
    eval(Expr::FunctionCall {
      name: "Expand".to_string(),
      args: vec![e.clone()].into(),
    })
    .unwrap_or_else(|_| e.clone())
  };
  let exponent = |e: &Expr| -> Option<i128> {
    match eval(Expr::FunctionCall {
      name: "Exponent".to_string(),
      args: vec![e.clone(), var_id.clone()].into(),
    }) {
      Ok(Expr::Integer(n)) => Some(n),
      _ => None,
    }
  };
  let coeff = |e: &Expr, d: i128| -> Expr {
    eval(Expr::FunctionCall {
      name: "Coefficient".to_string(),
      args: vec![e.clone(), var_id.clone(), Expr::Integer(d)].into(),
    })
    .unwrap_or(Expr::Integer(0))
  };
  let is_zero = |e: &Expr| matches!(e, Expr::Integer(0));
  let var_pow = |d: i128| -> Expr {
    if d == 0 {
      Expr::Integer(1)
    } else if d == 1 {
      var_id.clone()
    } else {
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![var_id.clone(), Expr::Integer(d)].into(),
      }
    }
  };

  // Precompute divisor leading data: (degree, leading_coeff, expanded_divisor).
  let mut div_info: Vec<Option<(i128, Expr, Expr)>> = Vec::new();
  for d in &divisors {
    let de = expand(d);
    if is_zero(&de) {
      div_info.push(None);
      continue;
    }
    match exponent(&de) {
      Some(deg) => {
        let lc = coeff(&de, deg);
        div_info.push(Some((deg, lc, de)));
      }
      None => return unevaluated(), // not a polynomial in `var`
    }
  }

  let k = divisors.len();
  let mut quotients = vec![Expr::Integer(0); k];
  let mut remainder = Expr::Integer(0);
  let mut p = expand(&args[0]);

  let mut guard = 0usize;
  while !is_zero(&p) {
    guard += 1;
    if guard > 100_000 {
      return unevaluated();
    }
    let dp = match exponent(&p) {
      Some(d) => d,
      None => return unevaluated(),
    };
    let lcp = coeff(&p, dp);

    // Find the first divisor whose leading term divides the leading term of p.
    let mut reduced = false;
    for (i, info) in div_info.iter().enumerate() {
      let Some((ddeg, dlc, de)) = info else {
        continue;
      };
      if *ddeg <= dp {
        // t = (lcp / dlc) * x^(dp - ddeg)
        let ratio = build_div(&lcp, dlc);
        let t = eval(build_mul(&ratio, &var_pow(dp - ddeg)))
          .unwrap_or_else(|_| build_mul(&ratio, &var_pow(dp - ddeg)));
        quotients[i] = eval(Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![quotients[i].clone(), t.clone()].into(),
        })
        .unwrap_or_else(|_| quotients[i].clone());
        p = expand(&build_sub(&p, &build_mul(&t, de)));
        reduced = true;
        break;
      }
    }
    if !reduced {
      // Move the leading term of p into the remainder.
      let lt = eval(build_mul(&lcp, &var_pow(dp)))
        .unwrap_or_else(|_| build_mul(&lcp, &var_pow(dp)));
      remainder = eval(Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![remainder.clone(), lt.clone()].into(),
      })
      .unwrap_or_else(|_| remainder.clone());
      p = expand(&build_sub(&p, &lt));
    }
  }

  Ok(Expr::List(
    vec![Expr::List(quotients.into()), remainder].into(),
  ))
}

/// Perform polynomial long division p / q in variable var.
/// Returns (quotient, remainder) as expressions.
pub fn poly_divide_symbolic(
  p: &Expr,
  q: &Expr,
  var: &str,
) -> Result<(Expr, Expr), InterpreterError> {
  // Get coefficients of both polynomials
  let p_expanded = expand_and_combine(p);
  let q_expanded = expand_and_combine(q);

  let p_deg = max_power_int(&p_expanded, var).unwrap_or(0);
  let q_deg = max_power_int(&q_expanded, var).unwrap_or(0);

  if q_deg == 0 {
    // Dividing by a constant - check if it's zero
    let q_coeff = coefficient_ast(&[
      q.clone(),
      Expr::Identifier(var.to_string()),
      Expr::Integer(0),
    ])?;
    let q_str =
      expr_to_string(&crate::evaluator::evaluate_expr_to_expr(&q_coeff)?);
    if q_str == "0" {
      return Err(InterpreterError::EvaluationError(
        "PolynomialRemainder: division by zero polynomial".into(),
      ));
    }
  }

  // Extract coefficients for p
  let mut p_coeffs: Vec<Expr> = Vec::new();
  for i in 0..=p_deg {
    let c = coefficient_ast(&[
      p.clone(),
      Expr::Identifier(var.to_string()),
      Expr::Integer(i),
    ])?;
    p_coeffs.push(crate::evaluator::evaluate_expr_to_expr(&c)?);
  }

  // Extract coefficients for q
  let mut q_coeffs: Vec<Expr> = Vec::new();
  for i in 0..=q_deg {
    let c = coefficient_ast(&[
      q.clone(),
      Expr::Identifier(var.to_string()),
      Expr::Integer(i),
    ])?;
    q_coeffs.push(crate::evaluator::evaluate_expr_to_expr(&c)?);
  }

  if p_deg < q_deg {
    // Remainder is p itself, quotient is 0
    return Ok((Expr::Integer(0), p_expanded));
  }

  // Polynomial long division using Expr arithmetic
  let mut remainder = p_coeffs;
  let mut quotient_coeffs =
    vec![Expr::Integer(0); (p_deg - q_deg + 1) as usize];
  let lead_q = q_coeffs.last().unwrap().clone();

  for i in (0..quotient_coeffs.len()).rev() {
    let rem_idx = i + q_coeffs.len() - 1;
    if rem_idx >= remainder.len() {
      continue;
    }

    // q_i = remainder[rem_idx] / lead_q
    let qi = build_div(&remainder[rem_idx], &lead_q);
    let qi = crate::evaluator::evaluate_expr_to_expr(&qi)?;

    quotient_coeffs[i] = qi.clone();

    // Subtract qi * q from remainder
    for j in 0..q_coeffs.len() {
      let sub = build_mul(&qi, &q_coeffs[j]);
      let sub = crate::evaluator::evaluate_expr_to_expr(&sub)?;
      let new_val = build_sub(&remainder[i + j], &sub);
      remainder[i + j] = crate::evaluator::evaluate_expr_to_expr(&new_val)?;
    }
  }

  // Build quotient expression
  let quotient = coeffs_to_expr_symbolic(&quotient_coeffs, var);
  let rem = coeffs_to_expr_symbolic(&remainder, var);

  Ok((quotient, rem))
}

/// Build a division expression
pub fn build_div(a: &Expr, b: &Expr) -> Expr {
  if expr_to_string(b) == "1" {
    return a.clone();
  }
  Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(a.clone()),
    right: Box::new(b.clone()),
  }
}

/// Build a multiplication expression
pub fn build_mul(a: &Expr, b: &Expr) -> Expr {
  Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![a.clone(), b.clone()].into(),
  }
}

/// Build a subtraction expression
pub fn build_sub(a: &Expr, b: &Expr) -> Expr {
  Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      a.clone(),
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), b.clone()].into(),
      },
    ]
    .into(),
  }
}

/// Build polynomial from symbolic coefficients
pub fn coeffs_to_expr_symbolic(coeffs: &[Expr], var: &str) -> Expr {
  let mut terms = Vec::new();
  for (i, coeff) in coeffs.iter().enumerate() {
    let c_str = expr_to_string(coeff);
    if c_str == "0" {
      continue;
    }
    let term = if i == 0 {
      coeff.clone()
    } else if i == 1 {
      if c_str == "1" {
        Expr::Identifier(var.to_string())
      } else {
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![coeff.clone(), Expr::Identifier(var.to_string())].into(),
        }
      }
    } else {
      let var_power = Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![Expr::Identifier(var.to_string()), Expr::Integer(i as i128)]
          .into(),
      };
      if c_str == "1" {
        var_power
      } else {
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![coeff.clone(), var_power].into(),
        }
      }
    };
    terms.push(term);
  }

  if terms.is_empty() {
    Expr::Integer(0)
  } else if terms.len() == 1 {
    terms.into_iter().next().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    }
  }
}
