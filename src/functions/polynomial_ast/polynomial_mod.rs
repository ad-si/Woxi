#[allow(unused_imports)]
use super::*;
use crate::evaluator::evaluate_expr_to_expr;

/// PolynomialMod[poly, m] — reduce poly modulo m.
///
/// A numeric modulus reduces the integer coefficients; a polynomial modulus
/// divides poly and keeps the remainder. A list of moduli reduces modulo all
/// of them at once, i.e. modulo the ideal they generate — so
/// `PolynomialMod[7 x^2 + 3, {x^2 - 1, 5}]` is 0 (the polynomial reduces to
/// 10, which vanishes mod 5) and `PolynomialMod[7, {5, 3}]` is 0 (the
/// integers generate everything).
pub fn polynomial_mod_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(unevaluated("PolynomialMod", args));
  }

  // PolynomialMod is Listable in its first argument.
  if let Expr::List(items) = &args[0] {
    let mapped: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| polynomial_mod_ast(&[item.clone(), args[1].clone()]))
      .collect();
    return Ok(Expr::List(mapped?.into()));
  }

  let moduli: Vec<Expr> = match &args[1] {
    Expr::List(items) => items.to_vec(),
    other => vec![other.clone()],
  };

  let mut numeric_gcd: i128 = 0;
  let mut polynomial_moduli: Vec<Expr> = Vec::new();
  for modulus in &moduli {
    let modulus = evaluate_expr_to_expr(modulus)?;
    match &modulus {
      // A zero modulus reduces nothing.
      Expr::Integer(0) => {}
      Expr::Integer(n) => numeric_gcd = gcd_i128(numeric_gcd, *n),
      _ => {
        if crate::functions::math_ast::expr_to_f64(&modulus).is_some() {
          // A non-integer numeric modulus is not supported.
          return Ok(unevaluated("PolynomialMod", args));
        }
        polynomial_moduli.push(modulus);
      }
    }
  }

  let mut result =
    evaluate_expr_to_expr(&super::expand_ast(&[args[0].clone()])?)?;

  // Divide out each polynomial modulus, in the variable it is written in.
  for modulus in &polynomial_moduli {
    let vars =
      crate::functions::math_ast::variables_ast(std::slice::from_ref(modulus))?;
    let Expr::List(var_list) = &vars else {
      return Ok(unevaluated("PolynomialMod", args));
    };
    let Some(var) = var_list.first() else {
      return Ok(unevaluated("PolynomialMod", args));
    };
    result = crate::evaluator::evaluate_function_call_ast(
      "PolynomialRemainder",
      &[result, modulus.clone(), var.clone()],
    )?;
  }

  if numeric_gcd == 1 {
    return Ok(Expr::Integer(0));
  }
  if numeric_gcd > 1 {
    result = reduce_coefficients(&result, numeric_gcd)?;
  }
  evaluate_expr_to_expr(&result)
}

/// Reduce every integer coefficient of an expanded polynomial modulo `m`,
/// taking the non-negative representative.
fn reduce_coefficients(expr: &Expr, m: i128) -> Result<Expr, InterpreterError> {
  let terms = collect_sum_terms(expr);
  let mut new_terms = Vec::new();
  for term in &terms {
    let (coeff, monomial) = extract_coefficient(term);
    let new_coeff = coeff.rem_euclid(m);
    if new_coeff != 0 {
      match monomial {
        Some(mon) => {
          if new_coeff == 1 {
            new_terms.push(mon);
          } else {
            new_terms.push(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(Expr::Integer(new_coeff)),
              right: Box::new(mon),
            });
          }
        }
        None => new_terms.push(Expr::Integer(new_coeff)),
      }
    }
  }

  if new_terms.is_empty() {
    return Ok(Expr::Integer(0));
  }
  let result = if new_terms.len() == 1 {
    new_terms.pop().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: new_terms.into(),
    }
  };
  evaluate_expr_to_expr(&result)
}

/// Flatten a sum expression into individual terms.
fn collect_sum_terms(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      let mut terms = collect_sum_terms(left);
      terms.extend(collect_sum_terms(right));
      terms
    }
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      let mut terms = collect_sum_terms(left);
      // Negate the right side terms
      for t in collect_sum_terms(right) {
        let (coeff, mon) = extract_coefficient(&t);
        match mon {
          Some(m) => terms.push(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(-coeff)),
            right: Box::new(m),
          }),
          None => terms.push(Expr::Integer(-coeff)),
        }
      }
      terms
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      args.iter().flat_map(collect_sum_terms).collect()
    }
    _ => vec![expr.clone()],
  }
}

/// Extract the integer coefficient and the monomial part from a term.
/// Returns (coefficient, Some(monomial)) or (coefficient, None) for pure numbers.
fn extract_coefficient(expr: &Expr) -> (i128, Option<Expr>) {
  match expr {
    Expr::Integer(n) => (*n, None),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if let Expr::Integer(n) = left.as_ref() {
        (*n, Some(*right.clone()))
      } else if let Expr::Integer(n) = right.as_ref() {
        (*n, Some(*left.clone()))
      } else {
        // No integer coefficient, implicit 1
        (1, Some(expr.clone()))
      }
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      // Look for an integer in the args
      for (i, arg) in args.iter().enumerate() {
        if let Expr::Integer(n) = arg {
          let remaining: Vec<Expr> = args
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, a)| a.clone())
            .collect();
          let monomial = if remaining.len() == 1 {
            remaining.into_iter().next().unwrap()
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: remaining.into(),
            }
          };
          return (*n, Some(monomial));
        }
      }
      // No integer factor, implicit coefficient 1
      (1, Some(expr.clone()))
    }
    // For identifiers, powers, etc. — implicit coefficient is 1
    _ => (1, Some(expr.clone())),
  }
}
