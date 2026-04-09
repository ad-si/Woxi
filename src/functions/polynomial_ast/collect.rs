#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr};

use crate::functions::calculus_ast::simplify;
use crate::functions::math_ast::times_ast;
use crate::functions::polynomial_ast::expand::is_sum;

// ─── Collect ────────────────────────────────────────────────────────

/// Collect[expr, x] - Collects terms by powers of x
pub fn collect_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Collect expects exactly 2 arguments".into(),
    ));
  }
  // Handle list of variables: Collect[expr, {x, y, ...}]
  // Collects by first variable, then recursively collects each coefficient by remaining.
  if let Expr::List(vars) = &args[1] {
    if vars.is_empty() {
      return Ok(args[0].clone());
    }
    if vars.len() == 1 {
      return collect_ast(&[args[0].clone(), vars[0].clone()]);
    }
    // Collect by first variable
    let first_collected = collect_ast(&[args[0].clone(), vars[0].clone()])?;
    // For each coefficient sub-expression, recursively collect by remaining vars
    let remaining = Expr::List(vars[1..].to_vec());
    return collect_in_coefficients(&first_collected, &vars[0], &remaining);
  }
  let var = match &args[1] {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Collect".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Expand and collect terms by power of var
  let expanded = expand_and_combine(&args[0]);
  let terms = collect_additive_terms(&expanded);

  // Group terms by power of var
  let mut power_groups: Vec<(i128, Vec<Expr>)> = Vec::new();

  for term in &terms {
    let (power, coeff) = term_var_power_and_coeff(term, var);
    if let Some(entry) = power_groups.iter_mut().find(|(p, _)| *p == power) {
      entry.1.push(coeff);
    } else {
      power_groups.push((power, vec![coeff]));
    }
  }

  // Sort by power ascending
  power_groups.sort_by_key(|(p, _)| *p);

  // Build result: sum of (combined_coeff * var^power)
  let mut result_terms = Vec::new();
  for (power, coeffs) in power_groups {
    // Sum the coefficients
    let combined_coeff = if coeffs.len() == 1 {
      simplify(coeffs[0].clone())
    } else {
      let sum = coeffs
        .into_iter()
        .reduce(|a, b| add_exprs(&a, &b))
        .unwrap_or(Expr::Integer(0));
      simplify(sum)
    };

    if matches!(&combined_coeff, Expr::Integer(0)) {
      continue;
    }

    let var_part = if power == 0 {
      None
    } else if power == 1 {
      Some(Expr::Identifier(var.to_string()))
    } else {
      Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::Identifier(var.to_string())),
        right: Box::new(Expr::Integer(power)),
      })
    };

    let term = match (combined_coeff.clone(), var_part) {
      (c, None) => c,
      (Expr::Integer(1), Some(v)) => v,
      (c, Some(v)) if matches!(&c, Expr::Integer(-1)) => negate_term(&v),
      (c, Some(v)) => {
        // If the coefficient is a sum, keep it grouped.
        // Wolfram ordering: if the coefficient contains a variable
        // that sorts after the collect variable, put var first.
        if is_sum(&c) {
          let mut coeff_vars = std::collections::HashSet::new();
          collect_variables(&c, &mut coeff_vars);
          if coeff_vars.iter().any(|cv| cv.as_str() > var) {
            multiply_exprs(&v, &c)
          } else {
            multiply_exprs(&c, &v)
          }
        } else {
          // Flatten coefficient and variable into a single product,
          // then use times_ast for Wolfram canonical ordering.
          let mut factors = Vec::new();
          flatten_product_factors_collect(&c, &mut factors);
          flatten_product_factors_collect(&v, &mut factors);
          times_ast(&factors).unwrap_or_else(|_| multiply_exprs(&c, &v))
        }
      }
    };
    result_terms.push(term);
  }

  if result_terms.is_empty() {
    Ok(Expr::Integer(0))
  } else {
    Ok(build_sum(result_terms))
  }
}

/// After collecting by `collected_var`, walk the result sum and apply
/// Collect[coeff, remaining_vars] to each grouped coefficient.
fn collect_in_coefficients(
  expr: &Expr,
  collected_var: &Expr,
  remaining_vars: &Expr,
) -> Result<Expr, InterpreterError> {
  let var_str = match collected_var {
    Expr::Identifier(s) => s.as_str(),
    _ => return Ok(expr.clone()),
  };

  // Re-expand to get individual terms for grouping
  let expanded = expand_and_combine(expr);
  let terms = collect_additive_terms(&expanded);

  // Group terms by power of the collected variable
  let mut power_groups: Vec<(i128, Vec<Expr>)> = Vec::new();
  for term in &terms {
    let (power, coeff) = term_var_power_and_coeff(term, var_str);
    if let Some(entry) = power_groups.iter_mut().find(|(p, _)| *p == power) {
      entry.1.push(coeff);
    } else {
      power_groups.push((power, vec![coeff]));
    }
  }

  power_groups.sort_by_key(|(p, _)| std::cmp::Reverse(*p));

  let mut result_terms = Vec::new();
  for (power, coeffs) in power_groups {
    // Sum the coefficients for this power
    let combined = if coeffs.len() == 1 {
      coeffs.into_iter().next().unwrap()
    } else {
      coeffs
        .into_iter()
        .reduce(|a, b| add_exprs(&a, &b))
        .unwrap_or(Expr::Integer(0))
    };

    // Recursively collect the combined coefficient by remaining variables
    let collected_coeff = collect_ast(&[combined, remaining_vars.clone()])?;

    if matches!(&collected_coeff, Expr::Integer(0)) {
      continue;
    }

    let var_part = if power == 0 {
      None
    } else if power == 1 {
      Some(collected_var.clone())
    } else {
      Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(collected_var.clone()),
        right: Box::new(Expr::Integer(power)),
      })
    };

    let rebuilt = match (collected_coeff, var_part) {
      (c, None) => c,
      (Expr::Integer(1), Some(v)) => v,
      (c, Some(v)) if matches!(&c, Expr::Integer(-1)) => negate_term(&v),
      (c, Some(v)) => {
        if is_sum(&c) {
          let mut coeff_vars = std::collections::HashSet::new();
          collect_variables(&c, &mut coeff_vars);
          if coeff_vars.iter().any(|cv| cv.as_str() > var_str) {
            multiply_exprs(&v, &c)
          } else {
            multiply_exprs(&c, &v)
          }
        } else {
          let mut factors = Vec::new();
          flatten_product_factors_collect(&c, &mut factors);
          flatten_product_factors_collect(&v, &mut factors);
          times_ast(&factors).unwrap_or_else(|_| multiply_exprs(&c, &v))
        }
      }
    };
    result_terms.push(rebuilt);
  }

  if result_terms.is_empty() {
    Ok(Expr::Integer(0))
  } else {
    Ok(build_sum(result_terms))
  }
}

/// Flatten a Times expression into its factors.
fn flatten_product_factors_collect(expr: &Expr, out: &mut Vec<Expr>) {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      flatten_product_factors_collect(left, out);
      flatten_product_factors_collect(right, out);
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      for a in args {
        flatten_product_factors_collect(a, out);
      }
    }
    _ => out.push(expr.clone()),
  }
}
