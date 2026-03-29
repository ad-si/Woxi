#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{Expr, expr_to_string};

/// PolynomialGCD[p1, p2, ...] - greatest common divisor of polynomials
pub fn polynomial_gcd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "PolynomialGCD expects at least 2 arguments".into(),
    ));
  }

  // Detect variables from all expressions
  let mut all_vars = std::collections::HashSet::new();
  for arg in args {
    collect_variables(arg, &mut all_vars);
  }

  // Remove known constants
  all_vars.remove("Pi");
  all_vars.remove("E");
  all_vars.remove("I");

  if all_vars.is_empty() {
    // All arguments are numeric - compute integer GCD
    return integer_gcd_multi(args);
  }

  // Sort variables alphabetically for deterministic behavior
  let mut var_list: Vec<_> = all_vars.into_iter().collect();
  var_list.sort();

  // For multivariate polynomials, use the first variable and recurse
  let var = &var_list[0];

  // Fold pairwise GCD
  let mut result = crate::evaluator::evaluate_expr_to_expr(&args[0])?;
  for arg in &args[1..] {
    let arg_eval = crate::evaluator::evaluate_expr_to_expr(arg)?;
    result = poly_gcd_pair(&result, &arg_eval, var)?;
  }

  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Compute GCD of two polynomials using the Euclidean algorithm
fn poly_gcd_pair(
  p: &Expr,
  q: &Expr,
  var: &str,
) -> Result<Expr, InterpreterError> {
  let p_str = expr_to_string(p);
  let q_str = expr_to_string(q);

  // Handle zero cases
  if p_str == "0" {
    return normalize_poly_sign(q, var);
  }
  if q_str == "0" {
    return normalize_poly_sign(p, var);
  }

  // Check degrees in the main variable
  let p_deg = max_power(&expand_and_combine(p), var).unwrap_or(0);
  let q_deg = max_power(&expand_and_combine(q), var).unwrap_or(0);

  if p_deg == 0 && q_deg == 0 {
    // Both are constants w.r.t. this variable - compute numeric GCD
    return integer_gcd_two(p, q);
  }

  // Extract integer content of both polynomials
  let p_content = poly_integer_content(p, var)?;
  let q_content = poly_integer_content(q, var)?;
  let content_gcd = integer_gcd_two(&p_content, &q_content)?;

  // Make both polynomials primitive (divide by content)
  let p_prim = poly_divide_by_constant(p, &p_content)?;
  let q_prim = poly_divide_by_constant(q, &q_content)?;

  // Euclidean algorithm on primitive parts
  let mut a = p_prim;
  let mut b = q_prim;

  for _ in 0..100 {
    let b_eval = crate::evaluator::evaluate_expr_to_expr(&b)?;
    let b_str = expr_to_string(&b_eval);
    if b_str == "0" {
      break;
    }

    // Try polynomial division; if it fails, the polynomials are coprime
    // (this handles multivariate cases where symbolic coefficients cause issues)
    match poly_divide_symbolic(&a, &b_eval, var) {
      Ok((_, remainder)) => {
        let remainder = crate::evaluator::evaluate_expr_to_expr(&remainder)?;
        a = b_eval;
        b = remainder;
      }
      Err(_) => {
        // Division failed - polynomials are likely coprime in this variable
        return Ok(content_gcd);
      }
    }
  }

  // Make result primitive with positive leading coefficient
  let a = crate::evaluator::evaluate_expr_to_expr(&a)?;

  // If the result has degree 0 in the main variable and is symbolic
  // (not a pure number), the polynomials are coprime in this variable
  let a_deg = max_power(&expand_and_combine(&a), var).unwrap_or(0);
  if a_deg == 0 && !is_numeric_expr(&a) {
    return Ok(content_gcd);
  }

  let a_prim = make_primitive(&a, var)?;

  // Multiply by the content GCD
  let content_str = expr_to_string(&content_gcd);
  if content_str == "1" {
    return Ok(a_prim);
  }

  let result = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![content_gcd, a_prim],
  };
  crate::evaluator::evaluate_expr_to_expr(&expand_and_combine(
    &crate::evaluator::evaluate_expr_to_expr(&result)?,
  ))
}

/// Extract the integer content (GCD of all numeric coefficients) of a polynomial.
/// For polynomials with symbolic coefficients, returns 1.
fn poly_integer_content(
  poly: &Expr,
  var: &str,
) -> Result<Expr, InterpreterError> {
  let expanded = expand_and_combine(poly);
  let deg = max_power(&expanded, var).unwrap_or(0);

  let mut int_coeffs = Vec::new();
  for i in 0..=deg {
    let c = coefficient_ast(&[
      poly.clone(),
      Expr::Identifier(var.to_string()),
      Expr::Integer(i),
    ])?;
    let c = crate::evaluator::evaluate_expr_to_expr(&c)?;
    let c_str = expr_to_string(&c);
    if c_str != "0" {
      // Only include if it's a pure integer or rational
      match &c {
        Expr::Integer(_) | Expr::BigInteger(_) => {
          let abs = Expr::FunctionCall {
            name: "Abs".to_string(),
            args: vec![c],
          };
          int_coeffs.push(crate::evaluator::evaluate_expr_to_expr(&abs)?);
        }
        Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          ..
        } => {
          // Rational number - include it
          let abs = Expr::FunctionCall {
            name: "Abs".to_string(),
            args: vec![c],
          };
          int_coeffs.push(crate::evaluator::evaluate_expr_to_expr(&abs)?);
        }
        _ => {
          // Symbolic coefficient - content is 1
          return Ok(Expr::Integer(1));
        }
      }
    }
  }

  if int_coeffs.is_empty() {
    return Ok(Expr::Integer(1));
  }
  if int_coeffs.len() == 1 {
    return Ok(int_coeffs.into_iter().next().unwrap());
  }

  let mut result = int_coeffs[0].clone();
  for c in &int_coeffs[1..] {
    result = integer_gcd_two(&result, c)?;
  }
  Ok(result)
}

/// Divide a polynomial by a constant
fn poly_divide_by_constant(
  poly: &Expr,
  constant: &Expr,
) -> Result<Expr, InterpreterError> {
  let c_str = expr_to_string(constant);
  if c_str == "1" {
    return Ok(poly.clone());
  }

  let div = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![constant.clone(), Expr::Integer(-1)],
      },
      poly.clone(),
    ],
  };
  let result = crate::evaluator::evaluate_expr_to_expr(&div)?;
  crate::evaluator::evaluate_expr_to_expr(&expand_and_combine(&result))
}

/// Make a polynomial primitive (content = 1) with positive leading coefficient
fn make_primitive(poly: &Expr, var: &str) -> Result<Expr, InterpreterError> {
  let content = poly_integer_content(poly, var)?;
  let prim = poly_divide_by_constant(poly, &content)?;
  normalize_poly_sign(&prim, var)
}

/// Ensure polynomial has positive leading coefficient
fn normalize_poly_sign(
  poly: &Expr,
  var: &str,
) -> Result<Expr, InterpreterError> {
  let expanded = expand_and_combine(poly);
  let deg = max_power(&expanded, var).unwrap_or(0);

  let lead_coeff = coefficient_ast(&[
    poly.clone(),
    Expr::Identifier(var.to_string()),
    Expr::Integer(deg),
  ])?;
  let lead_coeff = crate::evaluator::evaluate_expr_to_expr(&lead_coeff)?;

  let is_negative = match &lead_coeff {
    Expr::Integer(n) => *n < 0,
    Expr::Real(f) => *f < 0.0,
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left,
      ..
    } => matches!(left.as_ref(), Expr::Integer(n) if *n < 0),
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      ..
    } => true,
    _ => {
      let s = expr_to_string(&lead_coeff);
      s.starts_with('-')
    }
  };

  if is_negative {
    let neg = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), poly.clone()],
    };
    crate::evaluator::evaluate_expr_to_expr(&neg)
  } else {
    Ok(poly.clone())
  }
}

/// Check if an expression is purely numeric (no symbolic variables)
fn is_numeric_expr(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(_) | Expr::BigInteger(_) | Expr::Real(_) => true,
    Expr::BinaryOp { left, right, .. } => {
      is_numeric_expr(left) && is_numeric_expr(right)
    }
    Expr::UnaryOp { operand, .. } => is_numeric_expr(operand),
    Expr::FunctionCall { name, args } => {
      // Rational numbers like Times[-1, Power[...]] are numeric
      (name == "Times"
        || name == "Plus"
        || name == "Power"
        || name == "Rational")
        && args.iter().all(is_numeric_expr)
    }
    _ => false,
  }
}

/// Compute integer GCD of multiple values
fn integer_gcd_multi(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut result = args[0].clone();
  for arg in &args[1..] {
    result = integer_gcd_two(&result, arg)?;
  }
  Ok(result)
}

/// Compute GCD of two integer/rational expressions
fn integer_gcd_two(a: &Expr, b: &Expr) -> Result<Expr, InterpreterError> {
  let gcd_expr = Expr::FunctionCall {
    name: "GCD".to_string(),
    args: vec![a.clone(), b.clone()],
  };
  crate::evaluator::evaluate_expr_to_expr(&gcd_expr)
}
