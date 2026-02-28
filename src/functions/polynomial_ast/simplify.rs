#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator, expr_to_string};

use crate::functions::calculus_ast::simplify;

// ─── Simplify ───────────────────────────────────────────────────────

/// Simplify[expr] or Simplify[expr, Assumptions -> cond] - User-facing simplification
pub fn simplify_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Simplify expects 1 or 2 arguments".into(),
    ));
  }
  if args.len() == 2 {
    return simplify_with_assumptions(&args[0], &args[1], false);
  }
  Ok(simplify_expr(&args[0]))
}

/// FullSimplify[expr] or FullSimplify[expr, Assumptions -> cond]
pub fn full_simplify_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "FullSimplify expects 1 or 2 arguments".into(),
    ));
  }
  if args.len() == 2 {
    return simplify_with_assumptions(&args[0], &args[1], true);
  }
  // Thread over Lists
  if let Expr::List(items) = &args[0] {
    let results: Vec<Expr> = items.iter().map(full_simplify_expr).collect();
    return Ok(Expr::List(results));
  }
  Ok(full_simplify_expr(&args[0]))
}

/// Apply Simplify or FullSimplify with an Assumptions option.
fn simplify_with_assumptions(
  expr: &Expr,
  opts: &Expr,
  full: bool,
) -> Result<Expr, InterpreterError> {
  // Extract Assumptions -> value from the options argument
  let assumption = match opts {
    Expr::Rule {
      pattern,
      replacement,
    } => {
      if let Expr::Identifier(name) = pattern.as_ref() {
        if name == "Assumptions" {
          Some(replacement.as_ref().clone())
        } else {
          None
        }
      } else {
        None
      }
    }
    _ => None,
  };

  if let Some(assumption_val) = assumption {
    // Save previous $Assumptions
    let prev = crate::ENV.with(|e| e.borrow().get("$Assumptions").cloned());

    // Set $Assumptions
    let val = expr_to_string(&assumption_val);
    crate::ENV.with(|e| {
      e.borrow_mut()
        .insert("$Assumptions".to_string(), crate::StoredValue::Raw(val))
    });

    let result = if full {
      full_simplify_expr(expr)
    } else {
      simplify_expr(expr)
    };

    // Restore previous $Assumptions
    crate::ENV.with(|e| {
      let mut env = e.borrow_mut();
      if let Some(v) = prev {
        env.insert("$Assumptions".to_string(), v);
      } else {
        env.remove("$Assumptions");
      }
    });

    Ok(result)
  } else {
    // Unknown option, just simplify normally
    if full {
      Ok(full_simplify_expr(expr))
    } else {
      Ok(simplify_expr(expr))
    }
  }
}

/// FullSimplify: more aggressive than Simplify.
/// Expands, applies trig identities, factors out common terms, and tries factoring.
pub fn full_simplify_expr(expr: &Expr) -> Expr {
  // Thread over Lists
  if let Expr::List(items) = expr {
    let results: Vec<Expr> = items.iter().map(full_simplify_expr).collect();
    return Expr::List(results);
  }

  // First apply regular simplification
  let simplified = simplify_expr(expr);

  // Then expand fully and combine
  let expanded = expand_and_combine(&simplified);

  // Apply trig identities
  let trig_simplified = apply_trig_identities(&expanded);

  // Keep track of the best (simplest) form using leaf count as complexity
  let mut best = trig_simplified.clone();
  let mut best_complexity = leaf_count(&best);

  // Try factoring (Factor[expr]) — prefer factored forms (use <=)
  if let Ok(factored) =
    crate::functions::polynomial_ast::factor_ast(&[trig_simplified.clone()])
  {
    let c = leaf_count(&factored);
    if c <= best_complexity {
      best = factored;
      best_complexity = c;
    }
  }

  // Try FactorTerms to factor out common numeric/symbolic terms
  let terms = collect_additive_terms(&trig_simplified);
  if terms.len() >= 2 {
    if let Ok(factored) = crate::functions::polynomial_ast::factor_terms_ast(&[
      trig_simplified.clone(),
    ]) {
      let c = leaf_count(&factored);
      if c <= best_complexity {
        best = factored;
        best_complexity = c;
      }
    }

    // Try extracting common symbolic factors from all terms
    if let Some(factored) = factor_common_symbolic(&trig_simplified, &terms) {
      let c = leaf_count(&factored);
      if c <= best_complexity {
        best = factored;
        best_complexity = c;
      }
    }
  }

  let _ = best_complexity; // suppress unused warning
  best
}

/// Full simplification: expand, combine like terms, simplify.
pub fn simplify_expr(expr: &Expr) -> Expr {
  match expr {
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::String(_)
    | Expr::Constant(_)
    | Expr::Identifier(_) => expr.clone(),

    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      let num = simplify_expr(left);
      let den = simplify_expr(right);
      // Try to cancel: expand both and see if we can simplify
      simplify_division(&num, &den)
    }

    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      let base = simplify_expr(left);
      let exp = simplify_expr(right);
      simplify(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(base),
        right: Box::new(exp),
      })
    }

    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let l = simplify_expr(left);
      let r = simplify_expr(right);
      // Combine powers: x * x → x^2, x^a * x^b → x^(a+b)
      simplify_product(&l, &r)
    }

    Expr::BinaryOp {
      op: BinaryOperator::Plus | BinaryOperator::Minus,
      ..
    } => {
      let combined = expand_and_combine(expr);
      apply_trig_identities(&combined)
    }

    Expr::UnaryOp { op, operand } => {
      let inner = simplify_expr(operand);
      simplify(Expr::UnaryOp {
        op: *op,
        operand: Box::new(inner),
      })
    }

    // Handle FunctionCall forms of Plus, Times, Power
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Plus" => {
        let combined = expand_and_combine(expr);
        apply_trig_identities(&combined)
      }
      "Times" if args.len() == 2 => {
        let l = simplify_expr(&args[0]);
        let r = simplify_expr(&args[1]);
        simplify_product(&l, &r)
      }
      "Power" if args.len() == 2 => {
        let base = simplify_expr(&args[0]);
        let exp = simplify_expr(&args[1]);
        simplify(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(base),
          right: Box::new(exp),
        })
      }
      "Rational" if args.len() == 2 => expr.clone(),
      "ConditionalExpression" if args.len() == 2 => {
        simplify_conditional_expression(&args[0], &args[1])
      }
      _ => expr.clone(),
    },

    _ => simplify(expr.clone()),
  }
}

/// Simplify ConditionalExpression[value, cond] under current $Assumptions.
/// If cond matches $Assumptions → return Simplify[value]
/// If $Assumptions negates cond → return Undefined
/// Otherwise → ConditionalExpression[Simplify[value], cond]
pub fn simplify_conditional_expression(value: &Expr, cond: &Expr) -> Expr {
  let cond_str = expr_to_string(cond);

  // Get $Assumptions from environment (default: "True")
  let assumptions_str = crate::ENV
    .with(|e| {
      e.borrow().get("$Assumptions").map(|sv| match sv {
        crate::StoredValue::Raw(s) => s.clone(),
        crate::StoredValue::ExprVal(e) => expr_to_string(e),
        _ => "True".to_string(),
      })
    })
    .unwrap_or_else(|| "True".to_string());

  if cond_str == assumptions_str {
    // Condition matches assumptions → strip ConditionalExpression
    simplify_expr(value)
  } else if assumptions_str == format!("!{}", cond_str)
    || assumptions_str == format!(" !{}", cond_str)
    || assumptions_str == format!("Not[{}]", cond_str)
  {
    // Assumptions negate the condition → Undefined
    Expr::Identifier("Undefined".to_string())
  } else {
    // Keep ConditionalExpression with simplified value
    Expr::FunctionCall {
      name: "ConditionalExpression".to_string(),
      args: vec![simplify_expr(value), cond.clone()],
    }
  }
}

/// Apply trigonometric identities to a sum expression.
/// Detects a*Sin[x]^2 + a*Cos[x]^2 → a and similar patterns.
pub fn apply_trig_identities(expr: &Expr) -> Expr {
  let terms = collect_additive_terms(expr);
  if terms.len() < 2 {
    return expr.clone();
  }

  // Look for pairs: coeff*Sin[arg]^2 + coeff*Cos[arg]^2 → coeff
  let mut used = vec![false; terms.len()];
  let mut result_terms: Vec<Expr> = Vec::new();

  for i in 0..terms.len() {
    if used[i] {
      continue;
    }
    if let Some((coeff_i, arg_i, is_sin_i)) = extract_trig_squared(&terms[i]) {
      // Look for matching pair
      for j in (i + 1)..terms.len() {
        if used[j] {
          continue;
        }
        if let Some((coeff_j, arg_j, is_sin_j)) =
          extract_trig_squared(&terms[j])
          && is_sin_i != is_sin_j
          && expr_to_string(&arg_i) == expr_to_string(&arg_j)
          && expr_to_string(&coeff_i) == expr_to_string(&coeff_j)
        {
          // Found matching pair: coeff*Sin[x]^2 + coeff*Cos[x]^2 = coeff
          result_terms.push(coeff_i.clone());
          used[i] = true;
          used[j] = true;
          break;
        }
      }
    }
    if !used[i] {
      result_terms.push(terms[i].clone());
    }
  }

  if result_terms.len() == terms.len() {
    // No simplification happened
    return expr.clone();
  }

  // Re-combine to simplify (e.g. 1 + 1 → 2)
  if let Ok(result) = crate::functions::math_ast::plus_ast(&result_terms) {
    result
  } else {
    build_sum(result_terms)
  }
}

/// Try to extract (coefficient, argument, is_sin) from a term like coeff*Sin[arg]^2 or coeff*Cos[arg]^2.
pub fn extract_trig_squared(term: &Expr) -> Option<(Expr, Expr, bool)> {
  // Pattern: Sin[arg]^2 or Cos[arg]^2 (coefficient = 1)
  if let Some((func, arg)) = match_trig_squared(term) {
    return Some((Expr::Integer(1), arg, func == "Sin"));
  }

  // Pattern: coeff * Sin[arg]^2 or coeff * Cos[arg]^2
  let factors = collect_multiplicative_factors(term);
  if factors.len() < 2 {
    return None;
  }

  // Find the trig^2 factor
  for (idx, f) in factors.iter().enumerate() {
    if let Some((func, arg)) = match_trig_squared(f) {
      let mut coeff_factors: Vec<Expr> = Vec::new();
      for (j, g) in factors.iter().enumerate() {
        if j != idx {
          coeff_factors.push(g.clone());
        }
      }
      let coeff = if coeff_factors.len() == 1 {
        coeff_factors.remove(0)
      } else {
        build_product(coeff_factors)
      };
      return Some((coeff, arg, func == "Sin"));
    }
  }
  None
}

/// Match Sin[arg]^2 or Cos[arg]^2, returning ("Sin"/"Cos", arg).
pub fn match_trig_squared(expr: &Expr) -> Option<(&str, Expr)> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } if matches!(right.as_ref(), Expr::Integer(2)) => {
      if let Expr::FunctionCall { name, args } = left.as_ref()
        && (name == "Sin" || name == "Cos")
        && args.len() == 1
      {
        return Some((name.as_str(), args[0].clone()));
      }
      None
    }
    _ => None,
  }
}

/// Simplify a product, combining powers.
pub fn simplify_product(a: &Expr, b: &Expr) -> Expr {
  // x * x → x^2
  if expr_to_string(a) == expr_to_string(b) {
    return Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(a.clone()),
      right: Box::new(Expr::Integer(2)),
    };
  }

  // x^a * x^b → x^(a+b)
  let (base_a, exp_a) = extract_base_exp(a);
  let (base_b, exp_b) = extract_base_exp(b);
  if expr_to_string(&base_a) == expr_to_string(&base_b) {
    let new_exp = simplify(Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(exp_a),
      right: Box::new(exp_b),
    });
    return simplify(Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(base_a),
      right: Box::new(new_exp),
    });
  }

  simplify(Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(a.clone()),
    right: Box::new(b.clone()),
  })
}

/// Extract base and exponent from a power expression.
pub fn extract_base_exp(expr: &Expr) -> (Expr, Expr) {
  extract_base_and_exp(expr)
}

/// Simplify a division by trying polynomial cancellation.
pub fn simplify_division(num: &Expr, den: &Expr) -> Expr {
  // If same expression, return 1
  if expr_to_string(num) == expr_to_string(den) {
    return Expr::Integer(1);
  }

  // Try: if denominator is a single factor, try polynomial division
  // E.g. (x^2 - 1) / (x - 1) → x + 1
  // We try to do this by expanding the numerator and attempting polynomial long division
  let num_expanded = expand_and_combine(num);
  let den_expanded = expand_and_combine(den);

  // Try to find the variable
  if let Some(var) = find_single_variable(&num_expanded)
    && let Some(quotient) =
      poly_divide_single_var(&num_expanded, &den_expanded, &var)
  {
    return quotient;
  }

  simplify(Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(num_expanded),
    right: Box::new(den_expanded),
  })
}

/// Find a single variable in an expression (for univariate polynomial division).
pub fn find_single_variable(expr: &Expr) -> Option<String> {
  let mut vars = std::collections::HashSet::new();
  collect_variables(expr, &mut vars);
  if vars.len() == 1 {
    vars.into_iter().next()
  } else {
    None
  }
}

/// Collect all variable names from an expression.
pub(super) fn collect_variables(
  expr: &Expr,
  vars: &mut std::collections::HashSet<String>,
) {
  match expr {
    Expr::Identifier(name)
      if name != "True" && name != "False" && name != "Null" =>
    {
      vars.insert(name.clone());
    }
    Expr::BinaryOp { left, right, .. } => {
      collect_variables(left, vars);
      collect_variables(right, vars);
    }
    Expr::UnaryOp { operand, .. } => collect_variables(operand, vars),
    Expr::FunctionCall { args, .. } => {
      for a in args {
        collect_variables(a, vars);
      }
    }
    Expr::List(items) => {
      for i in items {
        collect_variables(i, vars);
      }
    }
    _ => {}
  }
}

/// Try polynomial long division of num/den in a single variable.
/// Returns Some(quotient) if den divides num exactly.
pub fn poly_divide_single_var(
  num: &Expr,
  den: &Expr,
  var: &str,
) -> Option<Expr> {
  let num_coeffs = extract_poly_coeffs(num, var)?;
  let den_coeffs = extract_poly_coeffs(den, var)?;

  if den_coeffs.is_empty() {
    return None;
  }

  let num_deg = num_coeffs.len() as i128 - 1;
  let den_deg = den_coeffs.len() as i128 - 1;

  if num_deg < den_deg {
    return None;
  }

  // Polynomial long division with integer/rational coefficients
  let mut remainder = num_coeffs.clone();
  let mut quotient = vec![0i128; (num_deg - den_deg + 1) as usize];
  let lead_den = *den_coeffs.last()?;

  if lead_den == 0 {
    return None;
  }

  for i in (0..quotient.len()).rev() {
    let rem_idx = i + den_coeffs.len() - 1;
    if rem_idx >= remainder.len() {
      continue;
    }
    if remainder[rem_idx] % lead_den != 0 {
      return None; // Not exactly divisible with integers
    }
    let q = remainder[rem_idx] / lead_den;
    quotient[i] = q;
    for j in 0..den_coeffs.len() {
      remainder[i + j] -= q * den_coeffs[j];
    }
  }

  // Check remainder is zero
  if remainder.iter().any(|&c| c != 0) {
    return None;
  }

  // Build quotient polynomial
  Some(coeffs_to_expr(&quotient, var))
}

/// Extract integer polynomial coefficients from expr, indexed by power.
/// coeffs[i] = coefficient of var^i
pub fn extract_poly_coeffs(expr: &Expr, var: &str) -> Option<Vec<i128>> {
  let terms = collect_additive_terms(expr);
  let mut max_pow: i128 = 0;
  let mut term_data: Vec<(i128, i128)> = Vec::new(); // (power, integer_coeff)

  for term in &terms {
    let (power, coeff) = term_var_power_and_coeff(term, var);
    if power < 0 {
      return None; // non-polynomial term
    }
    let int_coeff = match &simplify(coeff) {
      Expr::Integer(n) => *n,
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => {
        if let Expr::Integer(n) = operand.as_ref() {
          -n
        } else {
          return None;
        }
      }
      _ => return None, // non-integer coefficient
    };
    max_pow = max_pow.max(power);
    term_data.push((power, int_coeff));
  }

  let mut coeffs = vec![0i128; (max_pow + 1) as usize];
  for (power, c) in term_data {
    coeffs[power as usize] += c;
  }

  Some(coeffs)
}

/// Build a polynomial expression from integer coefficients.
/// coeffs[i] = coefficient of var^i
pub fn coeffs_to_expr(coeffs: &[i128], var: &str) -> Expr {
  let mut terms: Vec<Expr> = Vec::new();

  for (i, &c) in coeffs.iter().enumerate() {
    if c == 0 {
      continue;
    }
    let var_part = if i == 0 {
      None
    } else if i == 1 {
      Some(Expr::Identifier(var.to_string()))
    } else {
      Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::Identifier(var.to_string())),
        right: Box::new(Expr::Integer(i as i128)),
      })
    };

    let term = match (c, var_part) {
      (c, None) => Expr::Integer(c),
      (1, Some(v)) => v,
      (-1, Some(v)) => negate_term(&v),
      (c, Some(v)) => Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(c)),
        right: Box::new(v),
      },
    };
    terms.push(term);
  }

  if terms.is_empty() {
    Expr::Integer(0)
  } else {
    build_sum(terms)
  }
}

/// Count the complexity of an expression (leaf nodes + internal nodes).
/// Used as a metric for choosing the simplest form.
fn leaf_count(expr: &Expr) -> usize {
  match expr {
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::String(_)
    | Expr::Constant(_)
    | Expr::Identifier(_) => 1,
    Expr::BinaryOp { left, right, .. } => {
      1 + leaf_count(left) + leaf_count(right)
    }
    Expr::UnaryOp { operand, .. } => 1 + leaf_count(operand),
    Expr::FunctionCall { args, .. } => {
      1 + args.iter().map(leaf_count).sum::<usize>()
    }
    Expr::List(items) => items.iter().map(leaf_count).sum::<usize>().max(1),
    _ => 1,
  }
}

/// Factor out common symbolic factors from additive terms.
/// e.g., `2*a^2 - 2*a^2*Sin[theta]` → `2*a^2*(1 - Sin[theta])`
fn factor_common_symbolic(_expr: &Expr, terms: &[Expr]) -> Option<Expr> {
  if terms.len() < 2 {
    return None;
  }

  // For each term, get the set of multiplicative factor strings
  let term_factor_sets: Vec<Vec<(String, Expr)>> = terms
    .iter()
    .map(|t| {
      let factors = collect_multiplicative_factors(t);
      // Flatten negation but track it
      let mut result: Vec<(String, Expr)> = Vec::new();
      for f in &factors {
        match f {
          Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand,
          } => {
            let inner = collect_multiplicative_factors(operand);
            result.push(("-1".to_string(), Expr::Integer(-1)));
            for i in &inner {
              result.push((expr_to_string(i), i.clone()));
            }
          }
          _ => result.push((expr_to_string(f), f.clone())),
        }
      }
      result
    })
    .collect();

  // Find factors common to ALL terms (by string comparison)
  // Exclude integer factors (handled by factor_terms_numeric already)
  let first_factors: Vec<(String, Expr)> = term_factor_sets[0]
    .iter()
    .filter(|(s, _)| {
      // Skip pure integers and -1
      s != "-1" && s.parse::<i128>().is_err()
    })
    .cloned()
    .collect();

  let mut common_factors: Vec<(String, Expr)> = Vec::new();
  for (s, e) in &first_factors {
    if term_factor_sets[1..]
      .iter()
      .all(|tfs| tfs.iter().any(|(ts, _)| ts == s))
    {
      // Check for duplicates in common_factors
      if !common_factors.iter().any(|(cs, _)| cs == s) {
        common_factors.push((s.clone(), e.clone()));
      }
    }
  }

  if common_factors.is_empty() {
    return None;
  }

  // Remove common factors from each term
  let mut new_terms: Vec<Expr> = Vec::new();
  for tfs in &term_factor_sets {
    let mut remaining: Vec<Expr> = Vec::new();
    let mut used: Vec<bool> = vec![false; common_factors.len()];

    for (s, e) in tfs {
      let mut is_common = false;
      for (ci, (cs, _)) in common_factors.iter().enumerate() {
        if !used[ci] && s == cs {
          used[ci] = true;
          is_common = true;
          break;
        }
      }
      if !is_common {
        remaining.push(e.clone());
      }
    }

    if remaining.is_empty() {
      new_terms.push(Expr::Integer(1));
    } else if remaining.len() == 1 {
      new_terms.push(remaining.remove(0));
    } else {
      new_terms.push(build_product(remaining));
    }
  }

  // Build result: common_factor * (sum of new_terms)
  let common_expr = if common_factors.len() == 1 {
    common_factors[0].1.clone()
  } else {
    build_product(common_factors.into_iter().map(|(_, e)| e).collect())
  };

  let sum = expand_and_combine(&build_sum(new_terms));

  // Also try to factor numeric GCD from the inner sum
  let inner = if let Ok(factored) =
    crate::functions::polynomial_ast::factor_terms_ast(&[sum.clone()])
  {
    factored
  } else {
    sum
  };

  // Build result with proper ordering: numeric_coeff * symbolic_factors * (inner_sum)
  // Extract numeric factor from inner if present
  let (num_factor, remainder) = match &inner {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Integer(_)) => {
      (Some(*left.clone()), *right.clone())
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      // -expr → factor of -1
      match operand.as_ref() {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left,
          right,
        } if matches!(left.as_ref(), Expr::Integer(n) if *n > 0) => {
          if let Expr::Integer(n) = left.as_ref() {
            (Some(Expr::Integer(-n)), *right.clone())
          } else {
            (Some(Expr::Integer(-1)), *operand.clone())
          }
        }
        _ => (Some(Expr::Integer(-1)), *operand.clone()),
      }
    }
    _ => (None, inner),
  };

  // Check if we should negate the coefficient and inner sum to match canonical form.
  // Wolfram convention: the first symbolic (non-constant) term in the inner sum
  // should have a positive coefficient. If not, negate both.
  let (final_num, final_remainder) = {
    let inner_terms = collect_additive_terms(&remainder);
    // Find first non-constant term
    let first_symbolic =
      inner_terms.iter().find(|t| !matches!(t, Expr::Integer(_)));
    let should_negate = if let Some(sym_term) = first_symbolic {
      // Check if it has a negative leading coefficient
      matches!(
        sym_term,
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          ..
        }
      ) || matches!(sym_term, Expr::BinaryOp { op: BinaryOperator::Times, left, .. }
            if matches!(left.as_ref(), Expr::Integer(n) if *n < 0)
              || matches!(left.as_ref(), Expr::UnaryOp { op: UnaryOperator::Minus, .. }))
    } else {
      false
    };
    if should_negate {
      let negated_remainder = expand_and_combine(&negate_term(&remainder));
      let negated_num = num_factor
        .map(|nf| match nf {
          Expr::Integer(n) => Expr::Integer(-n),
          _ => negate_term(&nf),
        })
        .unwrap_or(Expr::Integer(-1));
      (Some(negated_num), negated_remainder)
    } else {
      (num_factor, remainder)
    }
  };

  let mut factors: Vec<Expr> = Vec::new();
  if let Some(nf) = final_num {
    factors.push(nf);
  }
  factors.push(common_expr);
  factors.push(final_remainder);

  Some(build_product(factors))
}
