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
fn expr_contains_identifier(expr: &Expr, name: &str) -> bool {
  match expr {
    Expr::Identifier(n) => n == name,
    Expr::BinaryOp { left, right, .. } => {
      expr_contains_identifier(left, name)
        || expr_contains_identifier(right, name)
    }
    Expr::UnaryOp { operand, .. } => expr_contains_identifier(operand, name),
    Expr::FunctionCall { args, .. } => {
      args.iter().any(|a| expr_contains_identifier(a, name))
    }
    Expr::List(items) => {
      items.iter().any(|a| expr_contains_identifier(a, name))
    }
    _ => false,
  }
}

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

  // Fast path for `Coefficient[expr, var, 0]` when expr doesn't mention
  // var: the whole expression is the coefficient of var^0, and Expanding
  // would needlessly change the user's canonical form (matches
  // wolframscript).
  if power == 0 && !expr_contains_identifier(&args[0], var) {
    return Ok(args[0].clone());
  }

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

/// Build a nested list of zeros with the given shape.
/// For an empty shape, returns a single Expr::Integer(0) (a scalar).
fn zero_array(shape: &[i128]) -> Expr {
  if shape.is_empty() {
    Expr::Integer(0)
  } else {
    let inner = zero_array(&shape[1..]);
    Expr::List(vec![inner; shape[0] as usize])
  }
}

/// Recursively extract coefficients for a sequence of variables, producing a
/// rectangular nested list with shape (d1+1, d2+1, …, dn+1). Powers past the
/// actual degree in a slice pad with zeros of the correct sub-shape.
fn coefficient_list_multi(
  poly: &Expr,
  vars: &[&str],
  shape: &[i128],
) -> Result<Expr, InterpreterError> {
  if vars.is_empty() {
    return Ok(poly.clone());
  }
  let mut items = Vec::with_capacity(shape[0] as usize);
  for power in 0..shape[0] {
    let coeff = coefficient_ast(&[
      poly.clone(),
      Expr::Identifier(vars[0].to_string()),
      Expr::Integer(power),
    ])?;
    let simplified = crate::evaluator::evaluate_expr_to_expr(&coeff)?;
    let sub = if vars.len() == 1 {
      simplified
    } else if matches!(&simplified, Expr::Integer(n) if *n == 0) {
      // Slice is identically zero in the remaining variables: fill with
      // zeros of the correct shape instead of recursing.
      zero_array(&shape[1..])
    } else {
      coefficient_list_multi(&simplified, &vars[1..], &shape[1..])?
    };
    items.push(sub);
  }
  Ok(Expr::List(items))
}

/// CoefficientList[poly, var] - list of coefficients from power 0 to degree.
/// CoefficientList[poly, {x1, x2, …}] - rectangular nested list of coefficients
/// indexed by (x1^p1, x2^p2, …).
pub fn coefficient_list_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "CoefficientList expects 2 arguments".into(),
    ));
  }

  // SeriesData input: reduce via `Normal` first so the ordinary polynomial
  // path can extract the coefficients and trim trailing zeros.
  let normalized = match &args[0] {
    Expr::FunctionCall { name, .. } if name == "SeriesData" => {
      crate::evaluator::evaluate_function_call_ast(
        "Normal",
        &[args[0].clone()],
      )?
    }
    _ => args[0].clone(),
  };

  // Multivariate form: CoefficientList[poly, {x1, x2, …}]
  if let Expr::List(var_items) = &args[1] {
    let mut var_names: Vec<&str> = Vec::with_capacity(var_items.len());
    for item in var_items {
      match item {
        Expr::Identifier(name) => var_names.push(name.as_str()),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "CoefficientList".to_string(),
            args: args.to_vec(),
          });
        }
      }
    }
    let expanded = expand_and_combine(&normalized);
    let mut shape: Vec<i128> = Vec::with_capacity(var_names.len());
    for v in &var_names {
      match max_power_int(&expanded, v) {
        Some(d) => shape.push(d + 1),
        None => {
          return Ok(Expr::FunctionCall {
            name: "CoefficientList".to_string(),
            args: args.to_vec(),
          });
        }
      }
    }
    return coefficient_list_multi(&expanded, &var_names, &shape);
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
  let expanded = expand_and_combine(&normalized);

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

  // Extract coefficient for each power from 0 to degree. Use `normalized`
  // (rather than the raw `args[0]`) so SeriesData inputs see the expanded
  // polynomial form instead of the opaque SeriesData head.
  let mut coeffs = Vec::new();
  for power in 0..=degree {
    let coeff = coefficient_ast(&[
      normalized.clone(),
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

// ─── CoefficientArrays ─────────────────────────────────────────────────

/// CoefficientArrays[poly, var] or CoefficientArrays[poly, {x, y, …}]
/// Returns `{c_0, c_1, …, c_d}` where `c_k` is a sparse rank-k tensor of
/// shape `[n, …, n]` (with `n` = number of variables) holding the
/// coefficients of all degree-k monomials at their sorted-index slot,
/// matching wolframscript's `SparseArray[Automatic, dims, 0, …]` shape.
pub fn coefficient_arrays_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "CoefficientArrays expects 2 arguments".into(),
    ));
  }
  let vars: Vec<String> = match &args[1] {
    Expr::List(items) => {
      let mut v = Vec::with_capacity(items.len());
      for item in items {
        match item {
          Expr::Identifier(name) => v.push(name.clone()),
          _ => {
            return Ok(Expr::FunctionCall {
              name: "CoefficientArrays".to_string(),
              args: args.to_vec(),
            });
          }
        }
      }
      v
    }
    Expr::Identifier(name) => vec![name.clone()],
    _ => {
      return Ok(Expr::FunctionCall {
        name: "CoefficientArrays".to_string(),
        args: args.to_vec(),
      });
    }
  };
  if vars.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "CoefficientArrays".to_string(),
      args: args.to_vec(),
    });
  }
  // Multi-polynomial form: `CoefficientArrays[{p1, p2, …}, vars]` returns
  // SparseArrays of shape `[m, n, …, n]` where `m` is the polynomial
  // count and the leading multi-index component is the polynomial index.
  if let Expr::List(polys) = &args[0] {
    return coefficient_arrays_multi(polys, &vars);
  }
  let n = vars.len();
  let expanded = super::expand::expand_and_combine(&args[0]);
  let expanded = crate::evaluator::evaluate_expr_to_expr(&expanded)?;
  let terms = collect_additive_terms(&expanded);
  // Each entry: (degree, sorted_indices_1based, coefficient_expr).
  let mut entries: Vec<(usize, Vec<usize>, Expr)> = Vec::new();
  let mut max_degree = 0usize;
  for term in &terms {
    let evaled = crate::evaluator::evaluate_expr_to_expr(term)?;
    let mut exponents: Vec<i128> = Vec::with_capacity(n);
    let mut remaining = evaled.clone();
    for var in &vars {
      let power = term_power_of_var(&remaining, var);
      exponents.push(power);
      let (_, coeff) = term_var_power_and_coeff(&remaining, var);
      remaining = crate::evaluator::evaluate_expr_to_expr(&coeff)?;
    }
    let degree: usize = exponents
      .iter()
      .map(|&e| if e < 0 { 0 } else { e as usize })
      .sum();
    if degree > max_degree {
      max_degree = degree;
    }
    let mut indices = Vec::with_capacity(degree);
    for (i, &p) in exponents.iter().enumerate() {
      let p = if p < 0 { 0 } else { p as usize };
      for _ in 0..p {
        indices.push(i + 1);
      }
    }
    indices.sort();
    entries.push((degree, indices, remaining));
  }
  // Group entries by degree. Sum coefficients for duplicate index lists.
  let mut by_degree: Vec<Vec<(Vec<usize>, Expr)>> =
    (0..=max_degree).map(|_| Vec::new()).collect();
  for (deg, idx, coef) in entries {
    let bucket = &mut by_degree[deg];
    if let Some(existing) = bucket.iter_mut().find(|(i, _)| *i == idx) {
      existing.1 = add_exprs(&existing.1, &coef);
    } else {
      bucket.push((idx, coef));
    }
  }
  let mut output: Vec<Expr> = Vec::with_capacity(max_degree + 1);
  // c_0: sum of constant terms (degree 0 → empty index list).
  let c0 = by_degree[0]
    .iter()
    .map(|(_, c)| c.clone())
    .reduce(|a, b| add_exprs(&a, &b))
    .unwrap_or(Expr::Integer(0));
  let c0 = crate::evaluator::evaluate_expr_to_expr(&c0)?;
  output.push(c0);
  for k in 1..=max_degree {
    let bucket = &by_degree[k];
    let non_zero: Vec<(Vec<usize>, Expr)> = bucket
      .iter()
      .filter(|(_, c)| !matches!(c, Expr::Integer(0)))
      .cloned()
      .collect();
    output.push(build_sparse_array_for_coefficients(n, k, &non_zero));
  }
  Ok(Expr::List(output))
}

/// Build a `SparseArray[Automatic, [n; k], 0, …]` expression from a list
/// of (sorted-index, coefficient) entries.
fn build_sparse_array_for_coefficients(
  n: usize,
  k: usize,
  entries: &[(Vec<usize>, Expr)],
) -> Expr {
  let dims_list = Expr::List(vec![Expr::Integer(n as i128); k]);
  // Empty payload — shape varies between 1D (rowPtr length 2) and ≥2D
  // (rowPtr length n+1) to match wolframscript's CSR-like layout.
  if entries.is_empty() {
    let row_ptr = if k == 1 {
      Expr::List(vec![Expr::Integer(0), Expr::Integer(0)])
    } else {
      Expr::List(vec![Expr::Integer(0); n + 1])
    };
    let inner = Expr::List(vec![
      Expr::Integer(1),
      Expr::List(vec![row_ptr, Expr::List(vec![])]),
      Expr::List(vec![]),
    ]);
    return Expr::FunctionCall {
      name: "SparseArray".to_string(),
      args: vec![
        Expr::Identifier("Automatic".to_string()),
        dims_list,
        Expr::Integer(0),
        inner,
      ],
    };
  }
  // Sort entries by index for deterministic CSR layout.
  let mut sorted_entries: Vec<(Vec<usize>, Expr)> = entries.to_vec();
  sorted_entries.sort_by(|a, b| a.0.cmp(&b.0));
  if k == 1 {
    // 1D: rowPtr is `{0, count}`. Each colIndex is a 1-tuple `{idx}`.
    let row_ptr = Expr::List(vec![
      Expr::Integer(0),
      Expr::Integer(sorted_entries.len() as i128),
    ]);
    let col_indices = Expr::List(
      sorted_entries
        .iter()
        .map(|(idx, _)| Expr::List(vec![Expr::Integer(idx[0] as i128)]))
        .collect(),
    );
    let values =
      Expr::List(sorted_entries.iter().map(|(_, c)| c.clone()).collect());
    let inner = Expr::List(vec![
      Expr::Integer(1),
      Expr::List(vec![row_ptr, col_indices]),
      values,
    ]);
    return Expr::FunctionCall {
      name: "SparseArray".to_string(),
      args: vec![
        Expr::Identifier("Automatic".to_string()),
        dims_list,
        Expr::Integer(0),
        inner,
      ],
    };
  }
  // k ≥ 2: rowPtr length n+1, colIndices are (k-1)-tuples.
  let mut row_counts = vec![0i128; n];
  let mut col_indices_list: Vec<Expr> =
    Vec::with_capacity(sorted_entries.len());
  let mut values_list: Vec<Expr> = Vec::with_capacity(sorted_entries.len());
  for (idx, c) in &sorted_entries {
    let row = idx[0] - 1;
    row_counts[row] += 1;
    let col_idx: Vec<Expr> =
      idx[1..].iter().map(|&i| Expr::Integer(i as i128)).collect();
    col_indices_list.push(Expr::List(col_idx));
    values_list.push(c.clone());
  }
  let mut row_ptr = vec![Expr::Integer(0)];
  let mut acc = 0i128;
  for c in row_counts {
    acc += c;
    row_ptr.push(Expr::Integer(acc));
  }
  let inner = Expr::List(vec![
    Expr::Integer(1),
    Expr::List(vec![Expr::List(row_ptr), Expr::List(col_indices_list)]),
    Expr::List(values_list),
  ]);
  Expr::FunctionCall {
    name: "SparseArray".to_string(),
    args: vec![
      Expr::Identifier("Automatic".to_string()),
      dims_list,
      Expr::Integer(0),
      inner,
    ],
  }
}

/// Multi-polynomial CoefficientArrays. Returns a list of SparseArrays
/// where `c_d` has shape `[m, n, …, n]` (with `d` copies of `n`, plus
/// the leading polynomial dimension of size `m`). The `c_0` SparseArray
/// has shape `[m]` and (matching wolframscript's quirk) wraps each
/// non-zero constant as an unevaluated `Plus[0, value]` so its surface
/// form is `0 + value` instead of just `value`.
fn coefficient_arrays_multi(
  polys: &[Expr],
  vars: &[String],
) -> Result<Expr, InterpreterError> {
  let m = polys.len();
  let n = vars.len();
  // For each poly, collect (degree, sorted_indices, coefficient).
  let mut per_poly: Vec<Vec<(usize, Vec<usize>, Expr)>> =
    Vec::with_capacity(m);
  let mut max_degree = 0usize;
  for poly in polys {
    let expanded = super::expand::expand_and_combine(poly);
    let expanded = crate::evaluator::evaluate_expr_to_expr(&expanded)?;
    let terms = collect_additive_terms(&expanded);
    let mut entries: Vec<(usize, Vec<usize>, Expr)> = Vec::new();
    for term in &terms {
      let evaled = crate::evaluator::evaluate_expr_to_expr(term)?;
      let mut exponents: Vec<i128> = Vec::with_capacity(n);
      let mut remaining = evaled.clone();
      for var in vars {
        let power = term_power_of_var(&remaining, var);
        exponents.push(power);
        let (_, coeff) = term_var_power_and_coeff(&remaining, var);
        remaining = crate::evaluator::evaluate_expr_to_expr(&coeff)?;
      }
      let degree: usize = exponents
        .iter()
        .map(|&e| if e < 0 { 0 } else { e as usize })
        .sum();
      if degree > max_degree {
        max_degree = degree;
      }
      let mut indices = Vec::with_capacity(degree);
      for (i, &p) in exponents.iter().enumerate() {
        let p = if p < 0 { 0 } else { p as usize };
        for _ in 0..p {
          indices.push(i + 1);
        }
      }
      indices.sort();
      entries.push((degree, indices, remaining));
    }
    // Sum coefficients sharing the same (degree, indices) — same as the
    // single-poly path but kept per-polynomial here.
    let mut grouped: Vec<(usize, Vec<usize>, Expr)> = Vec::new();
    for (deg, idx, coef) in entries {
      if let Some(existing) = grouped
        .iter_mut()
        .find(|(d, i, _)| *d == deg && *i == idx)
      {
        existing.2 = add_exprs(&existing.2, &coef);
      } else {
        grouped.push((deg, idx, coef));
      }
    }
    // Drop zero coefficients.
    grouped.retain(|(_, _, c)| !matches!(c, Expr::Integer(0)));
    per_poly.push(grouped);
  }
  let mut output: Vec<Expr> = Vec::with_capacity(max_degree + 1);
  for d in 0..=max_degree {
    output.push(build_sparse_array_multi(d, m, n, &per_poly));
  }
  Ok(Expr::List(output))
}

/// Build the degree-`d` SparseArray for the multi-polynomial form. The
/// shape is `[m]` for `d == 0` (just one entry per polynomial) and
/// `[m, n, …, n]` (with `d` extra copies of `n`) for `d >= 1`. The
/// per-row CSR layout walks the polynomials in order.
fn build_sparse_array_multi(
  d: usize,
  m: usize,
  n: usize,
  per_poly: &[Vec<(usize, Vec<usize>, Expr)>],
) -> Expr {
  // dims: [m] for d=0, [m, n, n, …] (d copies of n) for d >= 1.
  let mut dims_vec = vec![Expr::Integer(m as i128)];
  for _ in 0..d {
    dims_vec.push(Expr::Integer(n as i128));
  }
  let dims_list = Expr::List(dims_vec);
  let make_outer = |inner: Expr| Expr::FunctionCall {
    name: "SparseArray".to_string(),
    args: vec![
      Expr::Identifier("Automatic".to_string()),
      dims_list.clone(),
      Expr::Integer(0),
      inner,
    ],
  };
  // Collect this degree's entries from each poly, in poly order.
  let mut row_counts = vec![0i128; m];
  let mut col_indices_list: Vec<Expr> = Vec::new();
  let mut values_list: Vec<Expr> = Vec::new();
  for (i, entries) in per_poly.iter().enumerate() {
    for (deg, idx, coef) in entries {
      if *deg != d {
        continue;
      }
      row_counts[i] += 1;
      // For d=0, colIndices are 1-tuples `{poly_index}` matching the
      // single-dim CSR. For d>=1, colIndices are `(d)`-tuples carrying
      // just the variable indices.
      let col: Vec<Expr> = if d == 0 {
        vec![Expr::Integer((i + 1) as i128)]
      } else {
        idx.iter().map(|&j| Expr::Integer(j as i128)).collect()
      };
      col_indices_list.push(Expr::List(col));
      // c_0's stored value is `Plus[0, coef]` — wolframscript's quirk
      // surfaces this accumulator as `0 + value` in the printed form.
      let value = if d == 0 {
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![Expr::Integer(0), coef.clone()],
        }
      } else {
        coef.clone()
      };
      values_list.push(value);
    }
  }
  // rowPtr length: `2` for d == 0 (single-row CSR), `m + 1` otherwise.
  let row_ptr = if d == 0 {
    let total: i128 = row_counts.iter().sum();
    Expr::List(vec![Expr::Integer(0), Expr::Integer(total)])
  } else {
    let mut v = vec![Expr::Integer(0)];
    let mut acc = 0i128;
    for c in &row_counts {
      acc += c;
      v.push(Expr::Integer(acc));
    }
    Expr::List(v)
  };
  let inner = Expr::List(vec![
    Expr::Integer(1),
    Expr::List(vec![row_ptr, Expr::List(col_indices_list)]),
    Expr::List(values_list),
  ]);
  make_outer(inner)
}
