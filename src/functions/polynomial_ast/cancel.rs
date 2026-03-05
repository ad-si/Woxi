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

      // Try multivariate cancellation: extract common monomial factors from
      // additive terms in the numerator, then cancel with the denominator.
      // E.g. (a^2*x + a*x^2) / a^2 → factor out a from numerator → a*(a*x + x^2) / a^2 → (a*x + x^2)/a
      {
        use super::coefficient::collect_additive_terms;
        use super::expand::{build_product, collect_multiplicative_factors};
        use crate::syntax::expr_to_string;

        let terms = collect_additive_terms(&num);
        if terms.len() >= 2 {
          // For each term, extract base→exp map of its multiplicative factors
          type BaseExpMap = Vec<(String, crate::syntax::Expr, i128)>;
          fn term_base_exp(term: &Expr) -> BaseExpMap {
            let factors = collect_multiplicative_factors(term);
            let mut map: BaseExpMap = Vec::new();
            for f in &factors {
              let (base_str, base, exp) = match f {
                Expr::BinaryOp {
                  op: BinaryOperator::Power,
                  left,
                  right,
                } => {
                  if let Expr::Integer(n) = right.as_ref() {
                    (expr_to_string(left), *left.clone(), *n)
                  } else {
                    (expr_to_string(f), f.clone(), 1)
                  }
                }
                Expr::FunctionCall { name, args }
                  if name == "Power" && args.len() == 2 =>
                {
                  if let Expr::Integer(n) = &args[1] {
                    (expr_to_string(&args[0]), args[0].clone(), *n)
                  } else {
                    (expr_to_string(f), f.clone(), 1)
                  }
                }
                Expr::Integer(_) => continue,
                _ => (expr_to_string(f), f.clone(), 1),
              };
              if let Some(entry) =
                map.iter_mut().find(|(k, _, _)| *k == base_str)
              {
                entry.2 += exp;
              } else {
                map.push((base_str, base, exp));
              }
            }
            map
          }

          // Find common base^exp across all terms (min exponent for each base)
          let first_map = term_base_exp(&terms[0]);
          let mut common: BaseExpMap = first_map;
          for term in &terms[1..] {
            let tmap = term_base_exp(term);
            common.retain_mut(|(key, _, exp)| {
              if let Some(entry) = tmap.iter().find(|(k, _, _)| k == key) {
                *exp = (*exp).min(entry.2);
                *exp > 0
              } else {
                false
              }
            });
          }

          if !common.is_empty() {
            // Build the common factor
            let common_factors: Vec<Expr> = common
              .iter()
              .map(|(_, base, exp)| {
                if *exp == 1 {
                  base.clone()
                } else {
                  Expr::BinaryOp {
                    op: BinaryOperator::Power,
                    left: Box::new(base.clone()),
                    right: Box::new(Expr::Integer(*exp)),
                  }
                }
              })
              .collect();
            let common_factor = build_product(common_factors);

            // Now cancel common_factor with the denominator using symbolic factor cancellation
            let den_factors = collect_multiplicative_factors(&den);
            let cf_factors = collect_multiplicative_factors(&common_factor);

            // Build base→exp maps for common factor and denominator
            fn factor_base_exp(factors: &[Expr]) -> Vec<(String, Expr, i128)> {
              let mut map: Vec<(String, Expr, i128)> = Vec::new();
              for f in factors {
                let (base_str, base, exp) = match f {
                  Expr::BinaryOp {
                    op: BinaryOperator::Power,
                    left,
                    right,
                  } => {
                    if let Expr::Integer(n) = right.as_ref() {
                      (expr_to_string(left), *left.clone(), *n)
                    } else {
                      (expr_to_string(f), f.clone(), 1)
                    }
                  }
                  Expr::FunctionCall { name, args }
                    if name == "Power" && args.len() == 2 =>
                  {
                    if let Expr::Integer(n) = &args[1] {
                      (expr_to_string(&args[0]), args[0].clone(), *n)
                    } else {
                      (expr_to_string(f), f.clone(), 1)
                    }
                  }
                  Expr::Integer(_) => continue,
                  _ => (expr_to_string(f), f.clone(), 1),
                };
                if let Some(entry) =
                  map.iter_mut().find(|(k, _, _)| *k == base_str)
                {
                  entry.2 += exp;
                } else {
                  map.push((base_str, base, exp));
                }
              }
              map
            }

            let cf_map = factor_base_exp(&cf_factors);
            let mut den_map = factor_base_exp(&den_factors);

            // Cancel: subtract common factor exponents from denominator
            let mut cancelled_from_num: Vec<(String, i128)> = Vec::new();
            for (key, _, cf_exp) in &cf_map {
              if let Some(den_entry) =
                den_map.iter_mut().find(|(k, _, _)| k == key)
              {
                let cancel_exp = (*cf_exp).min(den_entry.2);
                if cancel_exp > 0 {
                  den_entry.2 -= cancel_exp;
                  cancelled_from_num.push((key.clone(), cancel_exp));
                }
              }
            }

            if !cancelled_from_num.is_empty() {
              // Rebuild numerator: divide each term by cancelled factors
              let mut new_terms = Vec::new();
              for term in &terms {
                let mut t_factors = collect_multiplicative_factors(term);
                for (cancel_key, cancel_exp) in &cancelled_from_num {
                  let mut remaining = *cancel_exp;
                  let mut new_factors = Vec::new();
                  for f in t_factors {
                    if remaining <= 0 {
                      new_factors.push(f);
                      continue;
                    }
                    let (base_str, base, exp) = match &f {
                      Expr::BinaryOp {
                        op: BinaryOperator::Power,
                        left,
                        right,
                      } => {
                        if let Expr::Integer(n) = right.as_ref() {
                          (expr_to_string(left), *left.clone(), *n)
                        } else {
                          (expr_to_string(&f), f.clone(), 1)
                        }
                      }
                      Expr::FunctionCall { name, args }
                        if name == "Power" && args.len() == 2 =>
                      {
                        if let Expr::Integer(n) = &args[1] {
                          (expr_to_string(&args[0]), args[0].clone(), *n)
                        } else {
                          (expr_to_string(&f), f.clone(), 1)
                        }
                      }
                      Expr::Integer(_) => {
                        new_factors.push(f);
                        continue;
                      }
                      _ => (expr_to_string(&f), f.clone(), 1),
                    };
                    if base_str == *cancel_key {
                      let reduce = remaining.min(exp);
                      remaining -= reduce;
                      let new_exp = exp - reduce;
                      if new_exp > 1 {
                        new_factors.push(Expr::BinaryOp {
                          op: BinaryOperator::Power,
                          left: Box::new(base),
                          right: Box::new(Expr::Integer(new_exp)),
                        });
                      } else if new_exp == 1 {
                        new_factors.push(base);
                      }
                      // new_exp == 0: factor is gone
                    } else {
                      new_factors.push(f);
                    }
                  }
                  t_factors = new_factors;
                }
                if t_factors.is_empty() {
                  new_terms.push(Expr::Integer(1));
                } else {
                  new_terms.push(build_product(t_factors));
                }
              }

              // Build new numerator as sum
              let new_num_expr = if new_terms.len() == 1 {
                expand_and_combine(&new_terms[0])
              } else {
                let sum = super::expand::build_sum(new_terms);
                expand_and_combine(&sum)
              };

              // Rebuild denominator
              let mut new_den_factors: Vec<Expr> = Vec::new();
              // Collect integer factors from original denominator
              for f in &den_factors {
                if let Expr::Integer(n) = f {
                  new_den_factors.push(Expr::Integer(*n));
                }
              }
              for (_, base, exp) in &den_map {
                if *exp > 1 {
                  new_den_factors.push(Expr::BinaryOp {
                    op: BinaryOperator::Power,
                    left: Box::new(base.clone()),
                    right: Box::new(Expr::Integer(*exp)),
                  });
                } else if *exp == 1 {
                  new_den_factors.push(base.clone());
                }
              }
              let new_den_expr = if new_den_factors.is_empty() {
                Expr::Integer(1)
              } else {
                build_product(new_den_factors)
              };

              if matches!(&new_den_expr, Expr::Integer(1)) {
                return new_num_expr;
              }
              // Recursively cancel in case more simplification is possible
              return cancel_expr(&Expr::BinaryOp {
                op: BinaryOperator::Divide,
                left: Box::new(new_num_expr),
                right: Box::new(new_den_expr),
              });
            }
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
/// Supports rational exponents: a^(3/2)/a^(1/2) → a
pub fn cancel_symbolic_factors(num: &Expr, den: &Expr) -> Expr {
  // Rational exponent as (numerator, denominator) pair
  type RatExp = (i128, i128);

  // Extract base string and rational exponent from a factor
  fn base_and_exp(f: &Expr) -> (String, RatExp) {
    match f {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right: exp,
      } => {
        let base_str = expr_to_string(left);
        match exp.as_ref() {
          Expr::Integer(n) => (base_str, (*n, 1)),
          Expr::FunctionCall { name, args }
            if name == "Rational" && args.len() == 2 =>
          {
            if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
              (base_str, (*n, *d))
            } else {
              (expr_to_string(f), (1, 1))
            }
          }
          Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: n,
            right: d,
          } => {
            if let (Expr::Integer(nv), Expr::Integer(dv)) =
              (n.as_ref(), d.as_ref())
            {
              (base_str, (*nv, *dv))
            } else {
              (expr_to_string(f), (1, 1))
            }
          }
          _ => (expr_to_string(f), (1, 1)),
        }
      }
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        let base_str = expr_to_string(&args[0]);
        match &args[1] {
          Expr::Integer(n) => (base_str, (*n, 1)),
          Expr::FunctionCall {
            name: rname,
            args: rargs,
          } if rname == "Rational" && rargs.len() == 2 => {
            if let (Expr::Integer(n), Expr::Integer(d)) = (&rargs[0], &rargs[1])
            {
              (base_str, (*n, *d))
            } else {
              (expr_to_string(f), (1, 1))
            }
          }
          _ => (expr_to_string(f), (1, 1)),
        }
      }
      // Sqrt[x] → base=x, exp=1/2
      Expr::FunctionCall { name, args }
        if name == "Sqrt" && args.len() == 1 =>
      {
        (expr_to_string(&args[0]), (1, 2))
      }
      _ => (expr_to_string(f), (1, 1)),
    }
  }

  // Extract the base expression (without exponent) from a factor
  fn base_expr(f: &Expr) -> Expr {
    match f {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => match right.as_ref() {
        Expr::Integer(_) => left.as_ref().clone(),
        Expr::FunctionCall { name, args }
          if name == "Rational" && args.len() == 2 =>
        {
          left.as_ref().clone()
        }
        Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: n,
          right: d,
        } if matches!(n.as_ref(), Expr::Integer(_))
          && matches!(d.as_ref(), Expr::Integer(_)) =>
        {
          left.as_ref().clone()
        }
        _ => f.clone(),
      },
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        match &args[1] {
          Expr::Integer(_) => args[0].clone(),
          Expr::FunctionCall {
            name: rname,
            args: rargs,
          } if rname == "Rational" && rargs.len() == 2 => args[0].clone(),
          _ => f.clone(),
        }
      }
      Expr::FunctionCall { name, args }
        if name == "Sqrt" && args.len() == 1 =>
      {
        args[0].clone()
      }
      _ => f.clone(),
    }
  }

  // Reconstruct a factor from base expr and rational exponent
  fn make_factor(base: &Expr, exp: RatExp) -> Option<Expr> {
    let (n, d) = exp;
    if n == 0 {
      None
    } else if d == 1 && n == 1 {
      Some(base.clone())
    } else if d == 1 {
      Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(base.clone()),
        right: Box::new(Expr::Integer(n)),
      })
    } else if n == 1 && d == 2 {
      // exp = 1/2: use Sqrt[base]
      Some(Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![base.clone()],
      })
    } else {
      // Rational exponent: base^Rational[n, d]
      Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(base.clone()),
        right: Box::new(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(n), Expr::Integer(d)],
        }),
      })
    }
  }

  // Subtract two rational exponents: (a/b) - (c/d)
  fn rat_sub(a: RatExp, b: RatExp) -> RatExp {
    let num = a.0 * b.1 - b.0 * a.1;
    let den = a.1 * b.1;
    let g = gcd_i128(num.abs(), den.abs());
    if g > 0 {
      (num / g, den / g)
    } else {
      (num, den)
    }
  }

  // Check if rational exponent is positive
  fn rat_positive(r: RatExp) -> bool {
    (r.0 > 0 && r.1 > 0) || (r.0 < 0 && r.1 < 0)
  }

  // Minimum of two positive rational exponents
  fn rat_min(a: RatExp, b: RatExp) -> RatExp {
    // Compare a/b with c/d: a*d vs c*b
    let lhs = a.0 * b.1;
    let rhs = b.0 * a.1;
    if lhs <= rhs { a } else { b }
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
  let mut numeric_changed = false;
  if num_coeff != 0 && den_coeff != 0 {
    let g = gcd_i128(num_coeff.abs(), den_coeff.abs());
    if g > 1 {
      num_coeff /= g;
      den_coeff /= g;
      numeric_changed = true;
    }
    // Keep signs normalized: negative in numerator
    if den_coeff < 0 {
      num_coeff = -num_coeff;
      den_coeff = -den_coeff;
    }
  }

  // Build maps of base → (original_expr, base_str, exponent) for numerator and denominator
  let mut num_map: Vec<(Expr, String, RatExp)> = num_factors
    .iter()
    .map(|f| {
      let (base_str, exp) = base_and_exp(f);
      let be = base_expr(f);
      (be, base_str, exp)
    })
    .collect();

  let mut den_map: Vec<(Expr, String, RatExp)> = den_factors
    .iter()
    .map(|f| {
      let (base_str, exp) = base_and_exp(f);
      let be = base_expr(f);
      (be, base_str, exp)
    })
    .collect();

  // Cancel common factors
  let mut changed = false;
  for (_, num_base_str, num_exp) in num_map.iter_mut() {
    for (_, den_base_str, den_exp) in den_map.iter_mut() {
      if *num_base_str == *den_base_str
        && rat_positive(*num_exp)
        && rat_positive(*den_exp)
      {
        let common = rat_min(*num_exp, *den_exp);
        *num_exp = rat_sub(*num_exp, common);
        *den_exp = rat_sub(*den_exp, common);
        changed = true;
      }
    }
  }

  if !changed && !numeric_changed {
    // Nothing was cancelled, return original
    return Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(num.clone()),
      right: Box::new(den.clone()),
    };
  }

  // Rebuild numerator and denominator
  let mut new_num_factors: Vec<Expr> = Vec::new();
  if num_coeff != 1 || (num_map.iter().all(|(_, _, e)| e.0 == 0)) {
    new_num_factors.push(Expr::Integer(num_coeff));
  }
  for (be, _, exp) in &num_map {
    if let Some(f) = make_factor(be, *exp) {
      new_num_factors.push(f);
    }
  }

  let mut new_den_factors: Vec<Expr> = Vec::new();
  if den_coeff != 1 || (den_map.iter().all(|(_, _, e)| e.0 == 0)) {
    new_den_factors.push(Expr::Integer(den_coeff));
  }
  for (be, _, exp) in &den_map {
    if let Some(f) = make_factor(be, *exp) {
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
