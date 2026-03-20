use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::syntax::{BinaryOperator, Expr};

/// PolynomialMod[poly, m] — reduce all integer coefficients in poly modulo m.
pub fn polynomial_mod_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "PolynomialMod".to_string(),
      args: args.to_vec(),
    });
  }

  let m = match &args[1] {
    Expr::Integer(n) if *n > 0 => *n,
    _ => {
      // Symbolic modulus: return the expanded polynomial
      let expanded = super::expand_ast(&[args[0].clone()])?;
      return evaluate_expr_to_expr(&expanded);
    }
  };

  // Expand the polynomial first
  let expanded = super::expand_ast(&[args[0].clone()])?;

  // Collect terms as a flat sum
  let terms = collect_sum_terms(&expanded);

  // For each term, extract its integer coefficient, apply mod, and reconstruct
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

  // Build the sum and simplify
  let result = if new_terms.len() == 1 {
    new_terms.pop().unwrap()
  } else {
    // Build a Plus expression
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: new_terms,
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
              args: remaining,
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
