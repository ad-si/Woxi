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
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Collect expects 2 or 3 arguments".into(),
    ));
  }

  // Extract optional head function (3rd argument)
  let head = if args.len() == 3 {
    Some(&args[2])
  } else {
    None
  };

  // Handle list of variables: Collect[expr, {x, y, ...}]
  // Collects by first variable, then recursively collects each coefficient by remaining.
  if let Expr::List(vars) = &args[1] {
    if vars.is_empty() {
      return Ok(args[0].clone());
    }
    if vars.len() == 1 {
      let mut call = vec![args[0].clone(), vars[0].clone()];
      if let Some(h) = head {
        call.push(h.clone());
      }
      return collect_ast(&call);
    }
    // Collect by first variable, then recursively by remaining
    // For multi-variable collect with head, apply head at the innermost level
    let first_call = vec![args[0].clone(), vars[0].clone()];
    let first_collected = collect_ast(&first_call)?;
    let remaining = Expr::List(vars[1..].to_vec());
    // Pass head to the inner collect via collect_in_coefficients
    if let Some(h) = head {
      return collect_in_coefficients_with_head(
        &first_collected,
        &vars[0],
        &remaining,
        h,
      );
    }
    return collect_in_coefficients(&first_collected, &vars[0], &remaining);
  }
  let var = match &args[1] {
    Expr::Identifier(name) => name.as_str(),
    Expr::FunctionCall { .. } => {
      // Compound target like q[x] or q[0, x]: replace it with a fresh
      // identifier, collect over the identifier, then substitute back.
      let placeholder = "__collect_target__";
      let placeholder_id = Expr::Identifier(placeholder.to_string());
      let substituted =
        substitute_expr_local(&args[0], &args[1], &placeholder_id);
      let mut sub_call = vec![substituted, placeholder_id.clone()];
      if let Some(h) = head {
        sub_call.push(h.clone());
      }
      let result = collect_ast(&sub_call)?;
      return Ok(substitute_expr_local(&result, &placeholder_id, &args[1]));
    }
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

    // Apply head function to coefficient if provided
    let combined_coeff = if let Some(h) = head {
      let wrapped = Expr::FunctionCall {
        name: if let Expr::Identifier(n) = h {
          n.clone()
        } else {
          crate::syntax::expr_to_string(h)
        },
        args: vec![combined_coeff],
      };
      crate::evaluator::evaluate_expr_to_expr(&wrapped).unwrap_or(wrapped)
    } else {
      combined_coeff
    };

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
  } else if head.is_some() {
    // Head-wrapped coefficients need Wolfram's canonical Plus ordering
    // (descending powers of the collect variable) rather than the ascending
    // "constant first" order used when the coefficient is a bare Plus.
    Ok(plus_ast_canonical(result_terms))
  } else {
    Ok(build_sum(result_terms))
  }
}

/// Run `terms` through the Plus evaluator to obtain canonical Plus ordering,
/// falling back to a plain Plus chain if evaluation fails.
fn plus_ast_canonical(terms: Vec<Expr>) -> Expr {
  match crate::functions::math_ast::plus_ast(&terms) {
    Ok(e) => e,
    Err(_) => build_sum(terms),
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

  // Sort by power ascending. The final canonical Plus order is applied
  // to `result_terms` below via `sort_collect_terms`, so the insertion
  // order here only determines tie-breaks when the monomial signatures
  // are equal.
  power_groups.sort_by_key(|(p, _)| *p);

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
        } else if let Some((plus_factor, other_factors)) = split_plus_factor(&c)
        {
          // Coefficient is a Times containing a Plus (e.g. `(a+b)*y`).
          // Preserve the Plus-first display order that Collect wants:
          //   Plus * x^power * other_factors.
          let mut factors = vec![plus_factor];
          factors.push(v);
          factors.extend(other_factors);
          build_times_chain(&factors)
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
    // Flatten any nested Plus (e.g. a coefficient that was itself a sum
    // from the recursive collect pass) so every leaf term gets its own
    // monomial signature and participates in the canonical Plus ordering.
    let mut flat_terms: Vec<Expr> = Vec::new();
    for t in &result_terms {
      flatten_plus_terms(t, &mut flat_terms);
    }
    if flat_terms.len() == 1 {
      Ok(flat_terms.into_iter().next().unwrap())
    } else {
      sort_collect_terms(&mut flat_terms);
      Ok(build_sum(flat_terms))
    }
  }
}

/// Recursively flatten Plus terms (both `BinaryOp::Plus` chains and
/// `FunctionCall["Plus"]`) into a flat vector of addends.
fn flatten_plus_terms(e: &Expr, out: &mut Vec<Expr>) {
  match e {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      flatten_plus_terms(left, out);
      flatten_plus_terms(right, out);
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      for a in args {
        flatten_plus_terms(a, out);
      }
    }
    _ => out.push(e.clone()),
  }
}

/// Sort Collect's result terms using Wolfram's canonical Plus ordering
/// over their monomial signatures.
///
/// Each term has the form `coeff * monomial` where `coeff` may be a numeric
/// factor or a `Plus[...]` (the collected coefficient) and `monomial` is a
/// product of the collect variables and possibly other identifiers. We sort
/// by a monomial signature — a sorted list of `(variable, exponent)` pairs
/// extracted from the non-`Plus`, non-numeric factors — comparing the
/// entries from the last (largest) variable backwards. Terms with shorter
/// signatures come first when all compared entries match.
fn sort_collect_terms(terms: &mut [Expr]) {
  terms.sort_by(|a, b| {
    let sig_a = monomial_signature(a);
    let sig_b = monomial_signature(b);
    compare_monomial_signatures(&sig_a, &sig_b)
  });
}

/// Return the monomial signature (sorted by variable name) for a Collect term.
fn monomial_signature(term: &Expr) -> Vec<(String, i128)> {
  let mut pairs: Vec<(String, i128)> = Vec::new();
  collect_monomial_factors(term, &mut pairs);
  pairs.sort_by(|(a, _), (b, _)| a.cmp(b));
  pairs
}

fn collect_monomial_factors(e: &Expr, out: &mut Vec<(String, i128)>) {
  match e {
    Expr::Identifier(name) => out.push((name.clone(), 1)),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      collect_monomial_factors(left, out);
      collect_monomial_factors(right, out);
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      for a in args {
        collect_monomial_factors(a, out);
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      if let (Expr::Identifier(name), Expr::Integer(exp)) =
        (left.as_ref(), right.as_ref())
      {
        out.push((name.clone(), *exp));
      }
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if let (Expr::Identifier(var), Expr::Integer(exp)) = (&args[0], &args[1])
      {
        out.push((var.clone(), *exp));
      }
    }
    // Plus coefficients, numbers, and other compound exprs contribute nothing.
    _ => {}
  }
}

/// Compare monomial signatures by Wolfram's canonical Plus order: walk
/// the entries from the end (largest variable) inward. When one signature
/// is a strict prefix of the other (after matching the largest entries),
/// the shorter one sorts first.
fn compare_monomial_signatures(
  a: &[(String, i128)],
  b: &[(String, i128)],
) -> std::cmp::Ordering {
  let mut ia = a.iter().rev();
  let mut ib = b.iter().rev();
  loop {
    match (ia.next(), ib.next()) {
      (Some((va, ea)), Some((vb, eb))) => {
        let name_cmp = va.cmp(vb);
        if name_cmp != std::cmp::Ordering::Equal {
          return name_cmp;
        }
        let exp_cmp = ea.cmp(eb);
        if exp_cmp != std::cmp::Ordering::Equal {
          return exp_cmp;
        }
      }
      (None, Some(_)) => return std::cmp::Ordering::Less,
      (Some(_), None) => return std::cmp::Ordering::Greater,
      (None, None) => return std::cmp::Ordering::Equal,
    }
  }
}

/// If `expr` is a Times expression with exactly one `Plus` factor, return
/// the Plus factor and the remaining (non-Plus) factors in their original
/// order. Otherwise return `None`.
fn split_plus_factor(expr: &Expr) -> Option<(Expr, Vec<Expr>)> {
  let mut factors = Vec::new();
  flatten_product_factors_collect(expr, &mut factors);
  let mut plus: Option<Expr> = None;
  let mut rest: Vec<Expr> = Vec::new();
  for f in factors {
    if is_sum(&f) {
      if plus.is_some() {
        return None; // more than one Plus factor — caller can fall back
      }
      plus = Some(f);
    } else {
      rest.push(f);
    }
  }
  plus.map(|p| (p, rest))
}

/// Build a Times expression by chaining the given factors with
/// `BinaryOp::Times`, preserving their order (so the caller controls the
/// display ordering). For a single factor, returns it directly.
fn build_times_chain(factors: &[Expr]) -> Expr {
  if factors.is_empty() {
    return Expr::Integer(1);
  }
  if factors.len() == 1 {
    return factors[0].clone();
  }
  let mut iter = factors.iter();
  let mut result = iter.next().unwrap().clone();
  for f in iter {
    result = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(result),
      right: Box::new(f.clone()),
    };
  }
  result
}

/// Recursively replace every occurrence of `from` with `to` in `expr`,
/// using structural equality (via pretty-printed form).
fn substitute_expr_local(expr: &Expr, from: &Expr, to: &Expr) -> Expr {
  if crate::syntax::expr_to_string(expr) == crate::syntax::expr_to_string(from)
  {
    return to.clone();
  }
  match expr {
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|e| substitute_expr_local(e, from, to))
        .collect(),
    ),
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|e| substitute_expr_local(e, from, to))
        .collect(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(substitute_expr_local(left, from, to)),
      right: Box::new(substitute_expr_local(right, from, to)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(substitute_expr_local(operand, from, to)),
    },
    _ => expr.clone(),
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

/// Like collect_in_coefficients but passes a head function to the inner collect calls
fn collect_in_coefficients_with_head(
  expr: &Expr,
  collected_var: &Expr,
  remaining_vars: &Expr,
  head: &Expr,
) -> Result<Expr, InterpreterError> {
  let var_str = match collected_var {
    Expr::Identifier(s) => s.as_str(),
    _ => return Ok(expr.clone()),
  };

  let expanded = expand_and_combine(expr);
  let terms = collect_additive_terms(&expanded);

  let mut power_groups: Vec<(i128, Vec<Expr>)> = Vec::new();
  for term in &terms {
    let (power, coeff) = term_var_power_and_coeff(term, var_str);
    if let Some(entry) = power_groups.iter_mut().find(|(p, _)| *p == power) {
      entry.1.push(coeff);
    } else {
      power_groups.push((power, vec![coeff]));
    }
  }

  power_groups.sort_by_key(|(p, _)| *p);

  let mut result_terms = Vec::new();
  for (power, coeffs) in power_groups {
    let combined = if coeffs.len() == 1 {
      coeffs.into_iter().next().unwrap()
    } else {
      coeffs
        .into_iter()
        .reduce(|a, b| add_exprs(&a, &b))
        .unwrap_or(Expr::Integer(0))
    };

    // Recursively collect with head
    let collected_coeff =
      collect_ast(&[combined, remaining_vars.clone(), head.clone()])?;

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
      (c, Some(v)) => multiply_exprs(&c, &v),
    };
    result_terms.push(rebuilt);
  }

  if result_terms.is_empty() {
    return Ok(Expr::Integer(0));
  }
  if result_terms.len() == 1 {
    return Ok(result_terms.into_iter().next().unwrap());
  }
  let result = result_terms
    .into_iter()
    .reduce(|a, b| add_exprs(&a, &b))
    .unwrap();
  Ok(result)
}
