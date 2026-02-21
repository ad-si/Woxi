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
        args: args.to_vec(),
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
        args: args.to_vec(),
      });
    }
  };

  let (quotient, _) = poly_divide_symbolic(&args[0], &args[1], var)?;
  crate::evaluator::evaluate_expr_to_expr(&quotient)
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

  let p_deg = max_power(&p_expanded, var).unwrap_or(0);
  let q_deg = max_power(&q_expanded, var).unwrap_or(0);

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
    args: vec![a.clone(), b.clone()],
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
        args: vec![Expr::Integer(-1), b.clone()],
      },
    ],
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
          args: vec![coeff.clone(), Expr::Identifier(var.to_string())],
        }
      }
    } else {
      let var_power = Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![Expr::Identifier(var.to_string()), Expr::Integer(i as i128)],
      };
      if c_str == "1" {
        var_power
      } else {
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![coeff.clone(), var_power],
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
      args: terms,
    }
  }
}
