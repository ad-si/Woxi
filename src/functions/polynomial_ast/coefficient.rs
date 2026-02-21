#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator};

use crate::functions::calculus_ast::{is_constant_wrt, simplify};

// ─── Coefficient ────────────────────────────────────────────────────

/// Coefficient[expr, var] or Coefficient[expr, var, n]
/// Returns the coefficient of var^n (default n=1).
pub fn coefficient_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Coefficient expects 2 or 3 arguments".into(),
    ));
  }
  let var = match &args[1] {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Second argument of Coefficient must be a symbol".into(),
      ));
    }
  };
  let power: i128 = if args.len() == 3 {
    match &args[2] {
      Expr::Integer(n) => *n,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Coefficient".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    1
  };

  // First expand, then extract coefficient
  let expanded = expand_expr(&args[0]);
  let terms = collect_additive_terms(&expanded);
  let mut coeff_sum: Vec<Expr> = Vec::new();

  for term in &terms {
    if let Some(c) = extract_coefficient_of_power(term, var, power) {
      coeff_sum.push(c);
    }
  }

  if coeff_sum.is_empty() {
    Ok(Expr::Integer(0))
  } else if coeff_sum.len() == 1 {
    Ok(coeff_sum.remove(0))
  } else {
    // Sum all coefficient contributions
    let mut result = coeff_sum.remove(0);
    for c in coeff_sum {
      result = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(result),
        right: Box::new(c),
      };
    }
    Ok(simplify(result))
  }
}

/// CoefficientList[poly, var] - list of coefficients from power 0 to degree.
pub fn coefficient_list_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "CoefficientList expects 2 arguments".into(),
    ));
  }
  let var = match &args[1] {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "CoefficientList".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Expand and combine the expression first
  let expanded = expand_and_combine(&args[0]);

  // Find the degree
  let degree = match max_power(&expanded, var) {
    Some(d) => d,
    None => {
      return Ok(Expr::FunctionCall {
        name: "CoefficientList".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Extract coefficient for each power from 0 to degree
  let mut coeffs = Vec::new();
  for power in 0..=degree {
    let coeff = coefficient_ast(&[
      args[0].clone(),
      args[1].clone(),
      Expr::Integer(power),
    ])?;
    let simplified = crate::evaluator::evaluate_expr_to_expr(&coeff)?;
    coeffs.push(simplified);
  }

  Ok(Expr::List(coeffs))
}

/// Collect all additive terms from an expression (flattening Plus).
pub fn collect_additive_terms(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      let mut terms = collect_additive_terms(left);
      terms.extend(collect_additive_terms(right));
      terms
    }
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      let mut terms = collect_additive_terms(left);
      let right_terms = collect_additive_terms(right);
      for t in right_terms {
        terms.push(Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(t),
        });
      }
      terms
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      let mut terms = Vec::new();
      for a in args {
        terms.extend(collect_additive_terms(a));
      }
      terms
    }
    _ => vec![expr.clone()],
  }
}

/// From a single term, extract the coefficient of `var^power`.
/// E.g. for term = 3*x^2, var = "x", power = 2 → Some(3)
/// For term = a*x^2, var = "x", power = 2 → Some(a)
pub fn extract_coefficient_of_power(
  term: &Expr,
  var: &str,
  power: i128,
) -> Option<Expr> {
  // Get the power of var and the remaining coefficient
  let (term_power, coeff) = term_var_power_and_coeff(term, var);
  if term_power == power {
    Some(coeff)
  } else {
    None
  }
}

/// Decompose a multiplicative term into (power_of_var, coefficient).
/// E.g. 3*x^2 → (2, 3); a*x → (1, a); 5 → (0, 5); x → (1, 1)
pub fn term_var_power_and_coeff(term: &Expr, var: &str) -> (i128, Expr) {
  match term {
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) | Expr::String(_) => {
      (0, term.clone())
    }
    Expr::Identifier(name) => {
      if name == var {
        (1, Expr::Integer(1))
      } else {
        (0, term.clone())
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      if let Expr::Identifier(name) = left.as_ref()
        && name == var
        && let Expr::Integer(n) = right.as_ref()
      {
        return (*n, Expr::Integer(1));
      }
      // Check if the whole thing is constant w.r.t. var
      if is_constant_wrt(term, var) {
        (0, term.clone())
      } else {
        // Complex expression — try to factor out var powers
        (-1, term.clone()) // sentinel: won't match any requested power
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let (lp, lc) = term_var_power_and_coeff(left, var);
      let (rp, rc) = term_var_power_and_coeff(right, var);
      let total_power = lp + rp;
      let coeff = multiply_exprs(&lc, &rc);
      (total_power, coeff)
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let (p, c) = term_var_power_and_coeff(operand, var);
      (
        p,
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(c),
        },
      )
    }
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Times" => {
        let mut total_power: i128 = 0;
        let mut coeffs: Vec<Expr> = Vec::new();
        for a in args {
          let (p, c) = term_var_power_and_coeff(a, var);
          total_power += p;
          coeffs.push(c);
        }
        let coeff = coeffs
          .into_iter()
          .reduce(|a, b| multiply_exprs(&a, &b))
          .unwrap_or(Expr::Integer(1));
        (total_power, coeff)
      }
      "Power" if args.len() == 2 => {
        if let Expr::Identifier(name) = &args[0]
          && name == var
          && let Expr::Integer(n) = &args[1]
        {
          return (*n, Expr::Integer(1));
        }
        if is_constant_wrt(term, var) {
          (0, term.clone())
        } else {
          (-1, term.clone())
        }
      }
      _ => {
        if is_constant_wrt(term, var) {
          (0, term.clone())
        } else {
          (-1, term.clone())
        }
      }
    },
    _ => {
      if is_constant_wrt(term, var) {
        (0, term.clone())
      } else {
        (-1, term.clone())
      }
    }
  }
}

/// Multiply two expressions, simplifying trivial cases.
pub fn multiply_exprs(a: &Expr, b: &Expr) -> Expr {
  match (a, b) {
    (Expr::Integer(1), _) => b.clone(),
    (_, Expr::Integer(1)) => a.clone(),
    (Expr::Integer(0), _) | (_, Expr::Integer(0)) => Expr::Integer(0),
    (Expr::Integer(x), Expr::Integer(y)) => Expr::Integer(x * y),
    _ => Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(a.clone()),
      right: Box::new(b.clone()),
    },
  }
}

/// Add two expressions, simplifying trivial cases.
pub fn add_exprs(a: &Expr, b: &Expr) -> Expr {
  match (a, b) {
    (Expr::Integer(0), _) => b.clone(),
    (_, Expr::Integer(0)) => a.clone(),
    (Expr::Integer(x), Expr::Integer(y)) => Expr::Integer(x + y),
    _ => Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(a.clone()),
      right: Box::new(b.clone()),
    },
  }
}
