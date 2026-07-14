#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{Expr, unevaluated};

/// True when `e` is a numeric literal equal to zero (integer or real).
fn is_zero_const(e: &Expr) -> bool {
  match e {
    Expr::Integer(0) => true,
    Expr::Real(r) => *r == 0.0,
    _ => false,
  }
}

/// Discriminant[poly, var] - polynomial discriminant
/// Disc(p, x) = (-1)^(n(n-1)/2) / a_n * Resultant(p, p', x)
pub fn discriminant_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // A `Modulus -> p` option reduces the computed discriminant's coefficients
  // modulo p (the discriminant is a polynomial in the input coefficients).
  if args.iter().any(|a| extract_modulus_option(a).is_some()) {
    let mut pos: Vec<Expr> = Vec::new();
    let mut modulus: Option<i128> = None;
    for a in args {
      if let Some(m) = extract_modulus_option(a) {
        modulus = Some(m);
      } else {
        pos.push(a.clone());
      }
    }
    if let Some(p) = modulus {
      let base = discriminant_ast(&pos)?;
      return super::resultant::reduce_coeffs_modulus(&base, p);
    }
  }

  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Discriminant expects 2 arguments".into(),
    ));
  }

  let poly = &args[0];
  let var = &args[1];

  let var_name = match var {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(unevaluated("Discriminant", args));
    }
  };

  // Get degree and leading coefficient
  let expanded = expand_and_combine(poly);
  let degree = match max_power_int(&expanded, var_name) {
    Some(d) => d,
    None => {
      return Ok(unevaluated("Discriminant", args));
    }
  };

  let leading_coeff = super::coefficient_ast(&[
    poly.clone(),
    var.clone(),
    Expr::Integer(degree),
  ])?;

  // Degree 0: the resultant formula does not apply (p' = 0). Following the
  // root-product form disc = (-1)^(n(n-1)/2) a_n^(2n-2) prod_{i<j}(r_i-r_j)^2,
  // the empty product leaves a_0^(2*0-2) = a_0^(-2); the zero polynomial is
  // special-cased to 0 (matching Wolfram).
  if degree == 0 {
    if is_zero_const(&leading_coeff) {
      return Ok(Expr::Integer(0));
    }
    let inv_sq = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![leading_coeff, Expr::Integer(-2)].into(),
    };
    let simplified = crate::evaluator::evaluate_expr_to_expr(&inv_sq)?;
    return match super::cancel_ast(&[simplified.clone()]) {
      Ok(c) => Ok(c),
      Err(_) => Ok(simplified),
    };
  }

  // Compute derivative p'(x)
  let dpoly =
    crate::functions::calculus_ast::differentiate_expr(poly, var_name)?;

  // Compute Resultant(p, p', x)
  let res = super::resultant_ast(&[poly.clone(), dpoly, var.clone()])?;

  // sign = (-1)^(n*(n-1)/2)
  let sign_exp = degree * (degree - 1) / 2;
  let sign = if sign_exp % 2 == 0 { 1 } else { -1 };

  // Result = sign * Resultant / a_n
  let signed_res = if sign == -1 {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), res].into(),
    }
  } else {
    res
  };

  let result = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      signed_res,
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![leading_coeff, Expr::Integer(-1)].into(),
      },
    ]
    .into(),
  };

  // Simplify the result
  let simplified = crate::evaluator::evaluate_expr_to_expr(&result)?;
  // Apply Cancel to simplify rational expressions
  let cancelled = super::cancel_ast(&[simplified.clone()]);
  match cancelled {
    Ok(c) => Ok(c),
    Err(_) => Ok(simplified),
  }
}
