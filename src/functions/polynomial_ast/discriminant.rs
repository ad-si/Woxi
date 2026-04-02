#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;

/// Discriminant[poly, var] - polynomial discriminant
/// Disc(p, x) = (-1)^(n(n-1)/2) / a_n * Resultant(p, p', x)
pub fn discriminant_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
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
      return Ok(Expr::FunctionCall {
        name: "Discriminant".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Compute derivative p'(x)
  let dpoly =
    crate::functions::calculus_ast::differentiate_expr(poly, var_name)?;

  // Compute Resultant(p, p', x)
  let res = super::resultant_ast(&[poly.clone(), dpoly, var.clone()])?;

  // Get degree and leading coefficient
  let expanded = expand_and_combine(poly);
  let degree = match max_power_int(&expanded, var_name) {
    Some(d) => d,
    None => {
      return Ok(Expr::FunctionCall {
        name: "Discriminant".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let leading_coeff = super::coefficient_ast(&[
    poly.clone(),
    var.clone(),
    Expr::Integer(degree),
  ])?;

  // sign = (-1)^(n*(n-1)/2)
  let sign_exp = degree * (degree - 1) / 2;
  let sign = if sign_exp % 2 == 0 { 1 } else { -1 };

  // Result = sign * Resultant / a_n
  let signed_res = if sign == -1 {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), res],
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
        args: vec![leading_coeff, Expr::Integer(-1)],
      },
    ],
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
