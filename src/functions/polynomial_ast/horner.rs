#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr};

use crate::functions::polynomial_ast::coefficient::coefficient_ast;
use crate::functions::polynomial_ast::exponent::max_power_int;
use crate::functions::polynomial_ast::simplify::collect_variables;

/// HornerForm[poly] or HornerForm[poly, var]
/// Converts a polynomial to Horner (nested) form.
pub fn horner_form_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "HornerForm expects 1 or 2 arguments".into(),
    ));
  }

  let expr = &args[0];

  // Determine the variable
  let var_name = if args.len() == 2 {
    // Explicit variable
    match &args[1] {
      Expr::Identifier(name) => name.clone(),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "HornerForm".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    // Auto-detect: for univariate, pick that variable;
    // for multivariate, pick the first alphabetically
    let mut vars = std::collections::HashSet::new();
    collect_variables(expr, &mut vars);
    if vars.is_empty() {
      // No variables: constant expression, return as-is
      return Ok(expr.clone());
    }
    if vars.len() == 1 {
      vars.into_iter().next().unwrap()
    } else {
      // Pick alphabetically first variable
      let mut sorted: Vec<String> = vars.into_iter().collect();
      sorted.sort();
      sorted.into_iter().next().unwrap()
    }
  };

  // Handle rational functions (divisions): apply HornerForm to num and denom
  if let Some((num, denom)) = extract_fraction(expr) {
    let horner_num = horner_form_expr(&num, &var_name)?;
    let horner_denom = horner_form_expr(&denom, &var_name)?;
    return Ok(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(horner_num),
      right: Box::new(horner_denom),
    });
  }

  horner_form_expr(expr, &var_name)
}

/// Apply Horner form to a polynomial expression with respect to `var`.
fn horner_form_expr(expr: &Expr, var: &str) -> Result<Expr, InterpreterError> {
  // First expand the expression
  let expanded = expand_and_combine(expr);

  // Find the degree of the polynomial in var
  let degree = match max_power_int(&expanded, var) {
    Some(d) if d >= 2 => d,
    _ => {
      // Degree 0 or 1: no nesting to do, return evaluated expression
      return crate::evaluator::evaluate_expr_to_expr(&expanded);
    }
  };

  // Extract coefficients for each power from 0 to degree
  let mut coeffs = Vec::new();
  for power in 0..=degree {
    let coeff = coefficient_ast(&[
      expanded.clone(),
      Expr::Identifier(var.to_string()),
      Expr::Integer(power),
    ])?;
    let simplified = crate::evaluator::evaluate_expr_to_expr(&coeff)?;
    coeffs.push(simplified);
  }

  // Build Horner form: c_0 + x * (c_1 + x * (c_2 + ... + x * c_n))
  // Start from the highest degree and work down
  let var_expr = Expr::Identifier(var.to_string());

  // Start with the highest coefficient
  let mut result = coeffs[degree as usize].clone();

  // Nest from degree-1 down to 0
  for i in (0..degree as usize).rev() {
    // result = c_i + var * result
    let times_part = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(var_expr.clone()),
      right: Box::new(result),
    };

    let coeff = &coeffs[i];
    if matches!(coeff, Expr::Integer(0)) {
      // c_i is 0, just use var * result
      result = times_part;
    } else {
      result = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(coeff.clone()),
        right: Box::new(times_part),
      };
    }
  }

  // Evaluate to simplify (e.g., 0 + x*... → x*..., etc.)
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Try to extract numerator and denominator from a fraction.
fn extract_fraction(expr: &Expr) -> Option<(Expr, Expr)> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => Some((*left.clone(), *right.clone())),
    Expr::FunctionCall { name, args } if name == "Times" => {
      // Look for Times[..., Power[denom, -1], ...]
      // Find the Power[_, -1] factor
      let mut denom_idx = None;
      for (i, arg) in args.iter().enumerate() {
        if let Expr::FunctionCall {
          name: pname,
          args: pargs,
        } = arg
          && pname == "Power"
          && pargs.len() == 2
          && matches!(&pargs[1], Expr::Integer(-1))
        {
          denom_idx = Some(i);
          break;
        }
        if let Expr::BinaryOp {
          op: BinaryOperator::Power,
          right: exp,
          ..
        } = arg
          && matches!(exp.as_ref(), Expr::Integer(-1))
        {
          denom_idx = Some(i);
          break;
        }
      }
      if let Some(di) = denom_idx {
        let denom = match &args[di] {
          Expr::FunctionCall { args: pargs, .. } => pargs[0].clone(),
          Expr::BinaryOp { left: base, .. } => *base.clone(),
          _ => unreachable!(),
        };
        // Build numerator from remaining factors
        let num_args: Vec<Expr> = args
          .iter()
          .enumerate()
          .filter(|(i, _)| *i != di)
          .map(|(_, a)| a.clone())
          .collect();
        let num = if num_args.len() == 1 {
          num_args.into_iter().next().unwrap()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: num_args,
          }
        };
        Some((num, denom))
      } else {
        None
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      // Check for expr * denom^(-1)
      if let Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: base,
        right: exp,
      } = right.as_ref()
        && matches!(exp.as_ref(), Expr::Integer(-1))
      {
        return Some((*left.clone(), *base.clone()));
      }
      if let Expr::FunctionCall {
        name: pname,
        args: pargs,
      } = right.as_ref()
        && pname == "Power"
        && pargs.len() == 2
        && matches!(&pargs[1], Expr::Integer(-1))
      {
        return Some((*left.clone(), pargs[0].clone()));
      }
      // Check left side too
      if let Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: base,
        right: exp,
      } = left.as_ref()
        && matches!(exp.as_ref(), Expr::Integer(-1))
      {
        return Some((*right.clone(), *base.clone()));
      }
      if let Expr::FunctionCall {
        name: pname,
        args: pargs,
      } = left.as_ref()
        && pname == "Power"
        && pargs.len() == 2
        && matches!(&pargs[1], Expr::Integer(-1))
      {
        return Some((*right.clone(), pargs[0].clone()));
      }
      None
    }
    _ => None,
  }
}
