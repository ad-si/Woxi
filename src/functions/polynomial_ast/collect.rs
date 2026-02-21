#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr};

use crate::functions::calculus_ast::simplify;

// ─── Collect ────────────────────────────────────────────────────────

/// Collect[expr, x] - Collects terms by powers of x
pub fn collect_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Collect expects exactly 2 arguments".into(),
    ));
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
        // Wolfram canonical Times ordering: coefficient goes first unless
        // the coefficient contains a variable that sorts after the collect
        // variable alphabetically, in which case the variable goes first.
        let mut coeff_vars = std::collections::HashSet::new();
        collect_variables(&c, &mut coeff_vars);
        let var_after = coeff_vars.iter().any(|cv| cv.as_str() > var);
        if var_after {
          multiply_exprs(&v, &c)
        } else {
          multiply_exprs(&c, &v)
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
