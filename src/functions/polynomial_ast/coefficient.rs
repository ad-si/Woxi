#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator};

use crate::functions::calculus_ast::{is_constant_wrt, simplify};

// ─── Coefficient ────────────────────────────────────────────────────

/// Coefficient[expr, form] / Coefficient[expr, var] / Coefficient[expr, var, n]
///
/// Three forms:
/// - Coefficient[expr, var]      → coefficient of var^1
/// - Coefficient[expr, var, n]   → coefficient of var^n
/// - Coefficient[expr, x^n]      → coefficient of x^n (monomial form)
pub fn coefficient_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Coefficient expects 2 or 3 arguments".into(),
    ));
  }

  // Monomial form: Coefficient[expr, x^n] → Coefficient[expr, x, n].
  // Only rewrite when no explicit exponent was passed as the 3rd arg.
  if args.len() == 2
    && let Some((inner_var, inner_pow)) = as_var_power(&args[1])
  {
    let rewritten = vec![
      args[0].clone(),
      Expr::Identifier(inner_var),
      Expr::Integer(inner_pow),
    ];
    return coefficient_ast(&rewritten);
  }

  // Multivariate monomial form: Coefficient[expr, x^a * y^b * ...] →
  // coefficient of the exact monomial across all listed variables.
  if args.len() == 2
    && let Some(var_powers) = as_multivar_monomial(&args[1])
    && var_powers.len() > 1
  {
    return coefficient_of_monomial(&args[0], &var_powers);
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

/// Decompose a product-of-variable-powers monomial (e.g. `x^2 * y^3 * z`)
/// into a list of (var, power) pairs. Returns `None` if any factor isn't a
/// variable or variable^integer. Single vars like `x` count as `x^1`.
fn as_multivar_monomial(expr: &Expr) -> Option<Vec<(String, i128)>> {
  let mut factors = Vec::new();
  collect_product_factors(expr, &mut factors);
  let mut result = Vec::new();
  for f in &factors {
    match f {
      Expr::Identifier(v) => result.push((v.clone(), 1)),
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => {
        if let (Expr::Identifier(v), Expr::Integer(n)) =
          (left.as_ref(), right.as_ref())
          && *n >= 1
        {
          result.push((v.clone(), *n));
        } else {
          return None;
        }
      }
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        if let (Expr::Identifier(v), Expr::Integer(n)) = (&args[0], &args[1])
          && *n >= 1
        {
          result.push((v.clone(), *n));
        } else {
          return None;
        }
      }
      _ => return None,
    }
  }
  if result.is_empty() {
    None
  } else {
    Some(result)
  }
}

fn collect_product_factors(expr: &Expr, out: &mut Vec<Expr>) {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      collect_product_factors(left, out);
      collect_product_factors(right, out);
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      for a in args {
        collect_product_factors(a, out);
      }
    }
    _ => out.push(expr.clone()),
  }
}

/// Extract the coefficient of a multivariate monomial from a polynomial.
fn coefficient_of_monomial(
  expr: &Expr,
  var_powers: &[(String, i128)],
) -> Result<Expr, InterpreterError> {
  let expanded = expand_expr(expr);
  let terms = collect_additive_terms(&expanded);
  let mut coeff_sum: Vec<Expr> = Vec::new();

  for term in &terms {
    // Peel off each requested variable's power from the term.
    let mut remaining = term.clone();
    let mut matches_all = true;
    for (var, want_power) in var_powers {
      let (p, c) = term_var_power_and_coeff(&remaining, var);
      if p != *want_power {
        matches_all = false;
        break;
      }
      remaining = c;
    }
    if matches_all {
      // The remaining factor must be constant w.r.t. all requested vars.
      let all_constant = var_powers
        .iter()
        .all(|(v, _)| is_constant_wrt(&remaining, v));
      if all_constant {
        coeff_sum.push(remaining);
      }
    }
  }

  if coeff_sum.is_empty() {
    Ok(Expr::Integer(0))
  } else if coeff_sum.len() == 1 {
    Ok(coeff_sum.remove(0))
  } else {
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

/// If `expr` is `x^n` for a plain symbol `x` and positive integer `n`,
/// return `Some((x, n))`. Used by Coefficient to accept monomial forms.
fn as_var_power(expr: &Expr) -> Option<(String, i128)> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      if let (Expr::Identifier(name), Expr::Integer(n)) =
        (left.as_ref(), right.as_ref())
        && *n >= 1
      {
        return Some((name.clone(), *n));
      }
      None
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if let (Expr::Identifier(v), Expr::Integer(n)) = (&args[0], &args[1])
        && *n >= 1
      {
        return Some((v.clone(), *n));
      }
      None
    }
    _ => None,
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
  let degree = match max_power_int(&expanded, var) {
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
        // Non-algebraic functions (Sin, Cos, etc.) are treated as coefficients
        // even if they contain the variable. Only Times/Power have polynomial
        // structure. This matches Wolfram's Coefficient behavior where
        // Coefficient[x*Cos[x], x] = Cos[x].
        (0, term.clone())
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

// ─── MonomialList ────────────────────────────────────────────────────

/// MonomialList[poly, {x, y, ...}] — list of monomials sorted by degree
pub fn monomial_list_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "MonomialList expects 2 arguments".into(),
    ));
  }

  // Extract variable names
  let vars: Vec<String> = match &args[1] {
    Expr::List(items) => items
      .iter()
      .filter_map(|item| {
        if let Expr::Identifier(name) = item {
          Some(name.clone())
        } else {
          None
        }
      })
      .collect(),
    Expr::Identifier(name) => vec![name.clone()],
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MonomialList".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Expand the polynomial
  let expanded = super::expand::expand_and_combine(&args[0]);
  let expanded = crate::evaluator::evaluate_expr_to_expr(&expanded)?;

  // Collect additive terms
  let terms = collect_additive_terms(&expanded);

  // For each term, compute the exponent vector
  let mut term_info: Vec<(Vec<i128>, Expr)> = Vec::new();
  for term in &terms {
    let evaled = crate::evaluator::evaluate_expr_to_expr(term)?;
    let mut exponents = Vec::new();
    for var in &vars {
      exponents.push(term_power_of_var(&evaled, var));
    }
    term_info.push((exponents, evaled));
  }

  // Sort by lexicographic order of exponent vectors (descending)
  term_info.sort_by(|a, b| {
    for (ea, eb) in a.0.iter().zip(b.0.iter()) {
      match eb.cmp(ea) {
        std::cmp::Ordering::Equal => continue,
        other => return other,
      }
    }
    std::cmp::Ordering::Equal
  });

  Ok(Expr::List(term_info.into_iter().map(|(_, t)| t).collect()))
}

/// Get the power of a single variable in a term
fn term_power_of_var(term: &Expr, var: &str) -> i128 {
  match term {
    Expr::Identifier(name) if name == var => 1,
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      if let Expr::Identifier(name) = left.as_ref()
        && name == var
        && let Expr::Integer(n) = right.as_ref()
      {
        return *n;
      }
      0
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let lp = term_power_of_var(left, var);
      let rp = term_power_of_var(right, var);
      lp + rp
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      args.iter().map(|a| term_power_of_var(a, var)).sum()
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if let Expr::Identifier(n) = &args[0]
        && n == var
        && let Expr::Integer(p) = &args[1]
      {
        return *p;
      }
      0
    }
    _ => 0,
  }
}

// ─── CoefficientRules ────────────────────────────────────────────────

/// CoefficientRules[poly, var] or CoefficientRules[poly, {x, y, ...}]
/// Returns a list of rules mapping exponent vectors to coefficients,
/// sorted in descending lexicographic order of the exponent vectors.
pub fn coefficient_rules_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "CoefficientRules expects 2 arguments".into(),
    ));
  }

  // Extract variable names
  let vars: Vec<String> = match &args[1] {
    Expr::List(items) => items
      .iter()
      .filter_map(|item| {
        if let Expr::Identifier(name) = item {
          Some(name.clone())
        } else {
          None
        }
      })
      .collect(),
    Expr::Identifier(name) => vec![name.clone()],
    _ => {
      return Ok(Expr::FunctionCall {
        name: "CoefficientRules".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if vars.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "CoefficientRules".to_string(),
      args: args.to_vec(),
    });
  }

  // Expand the polynomial
  let expanded = super::expand::expand_and_combine(&args[0]);
  let expanded = crate::evaluator::evaluate_expr_to_expr(&expanded)?;

  // Collect additive terms
  let terms = collect_additive_terms(&expanded);

  // For each term, compute the exponent vector and coefficient
  // We need to accumulate coefficients for identical exponent vectors
  let mut exponent_map: std::collections::BTreeMap<Vec<i128>, Vec<Expr>> =
    std::collections::BTreeMap::new();

  for term in &terms {
    let evaled = crate::evaluator::evaluate_expr_to_expr(term)?;

    // Extract the coefficient by peeling off all variable powers
    let mut exponents = Vec::new();
    let mut remaining = evaled.clone();

    for var in &vars {
      let power = term_power_of_var(&remaining, var);
      exponents.push(power);
      // Extract the coefficient with respect to this variable at this power
      let (_, coeff) = term_var_power_and_coeff(&remaining, var);
      remaining = crate::evaluator::evaluate_expr_to_expr(&coeff)?;
    }

    exponent_map.entry(exponents).or_default().push(remaining);
  }

  // Build the result: sort by descending lexicographic order
  let mut entries: Vec<(Vec<i128>, Expr)> = Vec::new();
  for (exps, coeffs) in exponent_map {
    // Sum all coefficients for this exponent vector
    let coeff = if coeffs.len() == 1 {
      coeffs.into_iter().next().unwrap()
    } else {
      let sum_expr =
        coeffs.into_iter().reduce(|a, b| add_exprs(&a, &b)).unwrap();
      crate::evaluator::evaluate_expr_to_expr(&sum_expr)?
    };

    // Skip zero coefficients
    if matches!(coeff, Expr::Integer(0)) {
      continue;
    }

    entries.push((exps, coeff));
  }

  // Sort descending lexicographic
  entries.sort_by(|a, b| {
    for (ea, eb) in a.0.iter().zip(b.0.iter()) {
      match eb.cmp(ea) {
        std::cmp::Ordering::Equal => continue,
        other => return other,
      }
    }
    std::cmp::Ordering::Equal
  });

  // Build the list of rules
  let rules: Vec<Expr> = entries
    .into_iter()
    .map(|(exps, coeff)| {
      let exp_list = Expr::List(exps.into_iter().map(Expr::Integer).collect());
      Expr::Rule {
        pattern: Box::new(exp_list),
        replacement: Box::new(coeff),
      }
    })
    .collect();

  // Return empty list for zero polynomial
  Ok(Expr::List(rules))
}
