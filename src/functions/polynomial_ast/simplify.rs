#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator, expr_to_string};

use crate::functions::calculus_ast::simplify;

// ─── Simplify ───────────────────────────────────────────────────────

/// Simplify[expr] - User-facing simplification
pub fn simplify_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Simplify expects exactly 1 argument".into(),
    ));
  }
  Ok(simplify_expr(&args[0]))
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
