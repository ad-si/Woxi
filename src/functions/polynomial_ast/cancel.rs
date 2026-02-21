#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, expr_to_string};

// ─── Cancel ─────────────────────────────────────────────────────────

/// Cancel[expr] - Cancels common factors between numerator and denominator
pub fn cancel_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Cancel expects exactly 1 argument".into(),
    ));
  }
  Ok(cancel_expr(&args[0]))
}

pub fn cancel_expr(expr: &Expr) -> Expr {
  // Look for division
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      let num = expand_and_combine(left);
      let den = expand_and_combine(right);

      // Try polynomial division
      if let Some(var) = find_single_variable_both(&num, &den)
        && let (Some(num_coeffs), Some(den_coeffs)) = (
          extract_poly_coeffs(&num, &var),
          extract_poly_coeffs(&den, &var),
        )
      {
        // Find the GCD of the two polynomials
        if let Some(gcd_coeffs) = poly_gcd(&num_coeffs, &den_coeffs)
          && (gcd_coeffs.len() > 1
            || (gcd_coeffs.len() == 1 && gcd_coeffs[0] != 1))
        {
          // Divide both by GCD
          if let (Some(mut new_num), Some(mut new_den)) = (
            poly_exact_divide(&num_coeffs, &gcd_coeffs),
            poly_exact_divide(&den_coeffs, &gcd_coeffs),
          ) {
            // Also cancel numeric content GCD (poly_gcd normalizes to
            // primitive, so numeric factors like gcd(2,4)=2 may remain)
            let num_content = new_num
              .iter()
              .copied()
              .filter(|&c| c != 0)
              .fold(0i128, gcd_i128);
            let den_content = new_den
              .iter()
              .copied()
              .filter(|&c| c != 0)
              .fold(0i128, gcd_i128);
            if num_content > 1 && den_content > 1 {
              let content_gcd = gcd_i128(num_content, den_content);
              if content_gcd > 1 {
                new_num = new_num.iter().map(|c| c / content_gcd).collect();
                new_den = new_den.iter().map(|c| c / content_gcd).collect();
              }
            }
            // Normalize sign: keep denominator positive
            if new_den.last().map(|&c| c < 0).unwrap_or(false) {
              new_num = new_num.iter().map(|c| -c).collect();
              new_den = new_den.iter().map(|c| -c).collect();
            }
            let num_expr = coeffs_to_expr(&new_num, &var);
            let den_expr = coeffs_to_expr(&new_den, &var);
            // If denominator is 1, just return numerator
            if new_den == [1] {
              return num_expr;
            }
            return Expr::BinaryOp {
              op: BinaryOperator::Divide,
              left: Box::new(num_expr),
              right: Box::new(den_expr),
            };
          }
        }
      }

      // Try symbolic factor cancellation for products (e.g. (a*b)/(a*c) → b/c)
      let result = cancel_symbolic_factors(&num, &den);
      if let Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: ref rl,
        right: ref rr,
      } = result
      {
        // Only accept if something actually changed
        if expr_to_string(rl) != expr_to_string(&num)
          || expr_to_string(rr) != expr_to_string(&den)
        {
          return result;
        }
      } else {
        // Result is not a division (fully cancelled), return it
        return result;
      }

      // Fall back to simplify_division
      simplify_division(&num, &den)
    }
    _ => expand_and_combine(expr),
  }
}

/// Cancel common symbolic factors between numerator and denominator.
/// E.g. (a*b)/(a*c) → b/c, (a^2*b)/(a*b^2) → a/b
pub fn cancel_symbolic_factors(num: &Expr, den: &Expr) -> Expr {
  // Extract base and exponent from a factor
  fn base_and_exp(f: &Expr) -> (String, i128) {
    match f {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right: exp,
      } => {
        let base_str = expr_to_string(left);
        if let Expr::Integer(n) = exp.as_ref() {
          (base_str, *n)
        } else {
          (expr_to_string(f), 1)
        }
      }
      _ => (expr_to_string(f), 1),
    }
  }

  // Reconstruct a factor from base expr and exponent
  fn make_factor(base: &Expr, exp: i128) -> Option<Expr> {
    if exp == 0 {
      None
    } else if exp == 1 {
      Some(base.clone())
    } else {
      Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(base.clone()),
        right: Box::new(Expr::Integer(exp)),
      })
    }
  }

  let mut num_factors = collect_multiplicative_factors(num);
  let mut den_factors = collect_multiplicative_factors(den);

  // Separate numeric coefficients from symbolic factors
  let mut num_coeff: i128 = 1;
  let mut den_coeff: i128 = 1;
  num_factors.retain(|f| {
    if let Expr::Integer(n) = f {
      num_coeff *= n;
      false
    } else {
      true
    }
  });
  den_factors.retain(|f| {
    if let Expr::Integer(n) = f {
      den_coeff *= n;
      false
    } else {
      true
    }
  });

  // Cancel numeric GCD
  if num_coeff != 0 && den_coeff != 0 {
    let g = gcd_i128(num_coeff.abs(), den_coeff.abs());
    if g > 1 {
      num_coeff /= g;
      den_coeff /= g;
    }
    // Keep signs normalized: negative in numerator
    if den_coeff < 0 {
      num_coeff = -num_coeff;
      den_coeff = -den_coeff;
    }
  }

  // Build maps of base → (original_expr, exponent) for numerator and denominator
  let mut num_map: Vec<(Expr, String, i128)> = num_factors
    .iter()
    .map(|f| {
      let (base_str, exp) = base_and_exp(f);
      // Find the base expression (without exponent)
      let base_expr = match f {
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left,
          right,
        } => {
          if let Expr::Integer(_) = right.as_ref() {
            left.as_ref().clone()
          } else {
            f.clone()
          }
        }
        _ => f.clone(),
      };
      (base_expr, base_str, exp)
    })
    .collect();

  let mut den_map: Vec<(Expr, String, i128)> = den_factors
    .iter()
    .map(|f| {
      let (base_str, exp) = base_and_exp(f);
      let base_expr = match f {
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left,
          right,
        } => {
          if let Expr::Integer(_) = right.as_ref() {
            left.as_ref().clone()
          } else {
            f.clone()
          }
        }
        _ => f.clone(),
      };
      (base_expr, base_str, exp)
    })
    .collect();

  // Cancel common factors
  let mut changed = false;
  for (_, num_base_str, num_exp) in num_map.iter_mut() {
    for (_, den_base_str, den_exp) in den_map.iter_mut() {
      if *num_base_str == *den_base_str && *num_exp > 0 && *den_exp > 0 {
        let common = (*num_exp).min(*den_exp);
        *num_exp -= common;
        *den_exp -= common;
        changed = true;
      }
    }
  }

  if !changed && num_coeff == 1 && den_coeff == 1 {
    // Nothing was cancelled, return original
    return Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(num.clone()),
      right: Box::new(den.clone()),
    };
  }

  // Rebuild numerator and denominator
  let mut new_num_factors: Vec<Expr> = Vec::new();
  if num_coeff != 1 || (num_map.iter().all(|(_, _, e)| *e == 0)) {
    new_num_factors.push(Expr::Integer(num_coeff));
  }
  for (base_expr, _, exp) in &num_map {
    if let Some(f) = make_factor(base_expr, *exp) {
      new_num_factors.push(f);
    }
  }

  let mut new_den_factors: Vec<Expr> = Vec::new();
  if den_coeff != 1 || (den_map.iter().all(|(_, _, e)| *e == 0)) {
    new_den_factors.push(Expr::Integer(den_coeff));
  }
  for (base_expr, _, exp) in &den_map {
    if let Some(f) = make_factor(base_expr, *exp) {
      new_den_factors.push(f);
    }
  }

  let new_num = build_product(new_num_factors);
  let new_den = build_product(new_den_factors);

  if let Expr::Integer(1) = &new_den {
    new_num
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(new_num),
      right: Box::new(new_den),
    }
  }
}

/// Find the single variable that appears in either or both expressions
pub fn find_single_variable_both(a: &Expr, b: &Expr) -> Option<String> {
  let mut vars = std::collections::HashSet::new();
  collect_variables(a, &mut vars);
  collect_variables(b, &mut vars);
  if vars.len() == 1 {
    vars.into_iter().next()
  } else {
    None
  }
}

/// Compute GCD of two integer polynomials using Euclidean algorithm
pub fn poly_gcd(a: &[i128], b: &[i128]) -> Option<Vec<i128>> {
  if b.iter().all(|&c| c == 0) {
    return Some(a.to_vec());
  }
  if a.iter().all(|&c| c == 0) {
    return Some(b.to_vec());
  }

  let mut r0 = a.to_vec();
  let mut r1 = b.to_vec();

  // Trim trailing zeros
  while r0.last() == Some(&0) && r0.len() > 1 {
    r0.pop();
  }
  while r1.last() == Some(&0) && r1.len() > 1 {
    r1.pop();
  }

  // Euclidean algorithm for polynomials
  for _ in 0..100 {
    if r1.iter().all(|&c| c == 0) || r1.is_empty() {
      // Normalize: make leading coefficient positive and divide by GCD of coefficients
      let g = r0.iter().copied().filter(|&c| c != 0).fold(0i128, gcd_i128);
      if g > 0 {
        let result: Vec<i128> = r0.iter().map(|c| c / g).collect();
        if result.last().map(|&c| c < 0).unwrap_or(false) {
          return Some(result.iter().map(|c| -c).collect());
        }
        return Some(result);
      }
      return Some(r0);
    }
    if r0.len() < r1.len() {
      std::mem::swap(&mut r0, &mut r1);
    }
    // Pseudo-remainder
    let remainder = poly_pseudo_remainder(&r0, &r1)?;
    r0 = r1;
    r1 = remainder;
  }
  None
}

/// Compute pseudo-remainder of polynomial division
pub fn poly_pseudo_remainder(a: &[i128], b: &[i128]) -> Option<Vec<i128>> {
  if b.is_empty() || b.iter().all(|&c| c == 0) {
    return None;
  }
  let mut rem = a.to_vec();
  let b_lead = *b.last()?;
  let b_deg = b.len() - 1;

  while rem.len() > b.len()
    || (rem.len() == b.len() && !rem.iter().all(|&c| c == 0))
  {
    while rem.last() == Some(&0) && rem.len() > 1 {
      rem.pop();
    }
    if rem.len() < b.len() {
      break;
    }
    let rem_lead = *rem.last()?;
    let rem_deg = rem.len() - 1;
    if rem_deg < b_deg {
      break;
    }
    let shift = rem_deg - b_deg;
    // Multiply remainder by b_lead and subtract rem_lead * shifted b
    for c in &mut rem {
      *c *= b_lead;
    }
    for (i, &bc) in b.iter().enumerate() {
      rem[i + shift] -= rem_lead * bc;
    }
    // Trim trailing zeros
    while rem.last() == Some(&0) && rem.len() > 1 {
      rem.pop();
    }
  }

  // Simplify by GCD of coefficients
  let g = rem
    .iter()
    .copied()
    .filter(|&c| c != 0)
    .fold(0i128, gcd_i128);
  if g > 1 {
    rem = rem.iter().map(|c| c / g).collect();
  }
  Some(rem)
}

/// Exact polynomial division: returns quotient if a is divisible by b
pub fn poly_exact_divide(a: &[i128], b: &[i128]) -> Option<Vec<i128>> {
  if b.is_empty() || b.iter().all(|&c| c == 0) {
    return None;
  }
  let a_deg = a.len() as i128 - 1;
  let b_deg = b.len() as i128 - 1;
  if a_deg < b_deg {
    return None;
  }
  let mut remainder = a.to_vec();
  let mut quotient = vec![0i128; (a_deg - b_deg + 1) as usize];
  let lead_b = *b.last()?;
  if lead_b == 0 {
    return None;
  }

  for i in (0..quotient.len()).rev() {
    let rem_idx = i + b.len() - 1;
    if rem_idx >= remainder.len() {
      continue;
    }
    if remainder[rem_idx] % lead_b != 0 {
      return None;
    }
    let q = remainder[rem_idx] / lead_b;
    quotient[i] = q;
    for j in 0..b.len() {
      remainder[i + j] -= q * b[j];
    }
  }

  if remainder.iter().any(|&c| c != 0) {
    return None;
  }
  Some(quotient)
}
