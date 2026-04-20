#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator};

// ─── Apart ──────────────────────────────────────────────────────────

/// Apart[expr] - Partial fraction decomposition
pub fn apart_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Apart expects 1 or 2 arguments".into(),
    ));
  }

  // Thread over lists
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| {
        if args.len() == 2 {
          apart_ast(&[item.clone(), args[1].clone()])
        } else {
          apart_ast(&[item.clone()])
        }
      })
      .collect();
    return Ok(Expr::List(results?));
  }

  // Short-circuit: if the expression has no denominator, Apart is a no-op.
  let (_num, den_check) = super::together::extract_num_den(&args[0]);
  if matches!(&den_check, Expr::Integer(1)) {
    return Ok(args[0].clone());
  }

  let var = if args.len() == 2 {
    match &args[1] {
      Expr::Identifier(name) => name.clone(),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Apart".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    // Find the variable automatically
    match find_single_variable(&args[0]) {
      Some(v) => v,
      None => {
        return Ok(Expr::FunctionCall {
          name: "Apart".to_string(),
          args: args.to_vec(),
        });
      }
    }
  };

  apart_expr(&args[0], &var)
}

pub fn apart_expr(expr: &Expr, var: &str) -> Result<Expr, InterpreterError> {
  // Extract numerator and denominator using the general-purpose extractor
  // which handles BinaryOp::Divide, Times[..., Power[..., -1]], etc.
  let (num, den) = super::together::extract_num_den(expr);
  if matches!(&den, Expr::Integer(1)) {
    // Not a fraction — return as-is
    return Ok(expr.clone());
  }

  // Rebuild the expression in Divide form for downstream functions
  let divide_expr = Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(num.clone()),
    right: Box::new(den.clone()),
  };

  let num_expanded = expand_and_combine(&num);
  let den_expanded = expand_and_combine(&den);

  // Try integer-coefficient approach first
  let num_coeffs = extract_poly_coeffs(&num_expanded, var);
  let den_coeffs = extract_poly_coeffs(&den_expanded, var);

  if let (Some(nc), Some(dc)) = (&num_coeffs, &den_coeffs) {
    // If numerator degree >= denominator degree, do polynomial division first
    if nc.len() >= dc.len() {
      if let Some(quot_coeffs) = poly_exact_divide(nc, dc) {
        return Ok(coeffs_to_expr(&quot_coeffs, var));
      }
      let (quotient, remainder) = poly_long_divide(nc, dc);
      if remainder.iter().all(|&c| c == 0) {
        return Ok(coeffs_to_expr(&quotient, var));
      }
      let quot_expr = coeffs_to_expr(&quotient, var);
      let rem_expr = coeffs_to_expr(&remainder, var);
      let frac = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(rem_expr),
        right: Box::new(den_expanded.clone()),
      };
      let apart_remainder = apart_proper_fraction(&frac, var)?;
      return Ok(add_exprs(&quot_expr, &apart_remainder));
    }

    return apart_proper_fraction(&divide_expr, var);
  }

  // Fall back to symbolic approach (multivariate case)
  apart_symbolic(&divide_expr, &num_expanded, &den, var)
}

/// Perform partial fraction decomposition for a proper fraction (deg(num) < deg(den))
pub fn apart_proper_fraction(
  expr: &Expr,
  var: &str,
) -> Result<Expr, InterpreterError> {
  let (num, den) = match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => (*left.clone(), *right.clone()),
    _ => return Ok(expr.clone()),
  };

  let den_expanded = expand_and_combine(&den);
  let den_coeffs = match extract_poly_coeffs(&den_expanded, var) {
    Some(c) => c,
    None => return Ok(expr.clone()),
  };

  // Factor the denominator
  let gcd_coeff = den_coeffs
    .iter()
    .copied()
    .filter(|&c| c != 0)
    .fold(0i128, gcd_i128);
  if gcd_coeff == 0 {
    return Ok(expr.clone());
  }
  let reduced: Vec<i128> = den_coeffs.iter().map(|c| c / gcd_coeff).collect();
  let (sign, reduced) = if reduced.last().map(|&c| c < 0).unwrap_or(false) {
    (-1i128, reduced.iter().map(|c| -c).collect::<Vec<_>>())
  } else {
    (1, reduced)
  };
  let _overall = gcd_coeff * sign;

  // Find roots of denominator
  let mut remaining = reduced.clone();
  let mut roots: Vec<i128> = Vec::new();

  loop {
    if remaining.len() <= 1 {
      break;
    }
    match find_integer_root(&remaining) {
      Some(root) => {
        roots.push(root);
        remaining = divide_by_root(&remaining, root);
      }
      None => break,
    }
  }

  if roots.len() < 2 {
    // Can't decompose further
    return Ok(expr.clone());
  }

  // Sort roots descending so linear factors (-root + x) appear in ascending order
  roots.sort_by(|a, b| b.cmp(a));

  // Simple partial fraction for distinct linear factors:
  // N(x) / ((x-r1)(x-r2)...) = A1/(x-r1) + A2/(x-r2) + ...
  // where Ai = N(ri) / product of (ri - rj) for j != i
  let num_expanded = expand_and_combine(&num);
  let num_coeffs = match extract_poly_coeffs(&num_expanded, var) {
    Some(c) => c,
    None => return Ok(expr.clone()),
  };

  // If there's a remaining irreducible factor, we can't do simple partial fractions
  if !remaining.iter().all(|&c| c == 0) && remaining.len() > 1 {
    return Ok(expr.clone());
  }

  let mut result_terms = Vec::new();
  let overall_factor = gcd_coeff * sign;

  for (i, &root) in roots.iter().enumerate() {
    let num_at_root = evaluate_poly(&num_coeffs, root);
    let mut den_product = 1i128;
    for (j, &other_root) in roots.iter().enumerate() {
      if i != j {
        den_product *= root - other_root;
      }
    }
    // Handle the remaining factor
    if remaining.len() == 1 && remaining[0] != 0 {
      den_product *= remaining[0];
    }
    den_product *= overall_factor;

    if den_product == 0 {
      return Ok(expr.clone());
    }

    // A_i = num_at_root / den_product
    // Term = A_i / (x - root) which in canonical form is A_i / (-root + x)
    let g = gcd_i128(num_at_root.abs(), den_product.abs());
    let (mut an, mut ad) = (num_at_root / g, den_product / g);
    if ad < 0 {
      an = -an;
      ad = -ad;
    }

    // Build linear factor: when root is 0, just use the variable directly
    let linear_factor = if root == 0 {
      Expr::Identifier(var.to_string())
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(-root)),
        right: Box::new(Expr::Identifier(var.to_string())),
      }
    };

    let abs_an = an.abs();
    let frac = if ad == 1 && abs_an == 1 {
      // Wolfram canonical form: (expr)^(-1)
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(linear_factor),
        right: Box::new(Expr::Integer(-1)),
      }
    } else if ad == 1 {
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Integer(abs_an)),
        right: Box::new(linear_factor),
      }
    } else {
      // an / (ad * linear_factor) — Wolfram format: 1/(2*(-1 + x))
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Integer(abs_an)),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(ad)),
          right: Box::new(linear_factor),
        }),
      }
    };

    if an < 0 {
      result_terms.push(Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(frac),
      });
    } else {
      result_terms.push(frac);
    }
  }

  if result_terms.is_empty() {
    Ok(expr.clone())
  } else {
    Ok(build_sum(result_terms))
  }
}

/// Polynomial long division returning (quotient, remainder) as coefficient vectors
pub fn poly_long_divide(num: &[i128], den: &[i128]) -> (Vec<i128>, Vec<i128>) {
  let n_deg = num.len();
  let d_deg = den.len();
  if n_deg < d_deg {
    return (vec![0], num.to_vec());
  }
  let mut remainder = num.to_vec();
  let mut quotient = vec![0i128; n_deg - d_deg + 1];
  let lead_den = den[d_deg - 1];
  if lead_den == 0 {
    return (vec![0], num.to_vec());
  }

  for i in (0..quotient.len()).rev() {
    let rem_idx = i + d_deg - 1;
    if rem_idx >= remainder.len() {
      continue;
    }
    if remainder[rem_idx] % lead_den != 0 {
      // Non-integer quotient - stop here
      return (vec![0], num.to_vec());
    }
    let q = remainder[rem_idx] / lead_den;
    quotient[i] = q;
    for j in 0..d_deg {
      remainder[i + j] -= q * den[j];
    }
  }
  (quotient, remainder)
}

// ─── Symbolic Apart (multivariate) ─────────────────────────────────

/// Flatten a product expression into its factors.
fn flatten_product_factors(expr: &Expr, out: &mut Vec<Expr>) {
  match expr {
    Expr::FunctionCall { name, args } if name == "Times" => {
      for a in args {
        flatten_product_factors(a, out);
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      flatten_product_factors(left, out);
      flatten_product_factors(right, out);
    }
    _ => out.push(expr.clone()),
  }
}

/// Check if expr is a polynomial of degree exactly 1 in var.
/// Returns Some((coeff_of_var, constant_term)) if linear.
fn extract_linear_coeffs(expr: &Expr, var: &str) -> Option<(Expr, Expr)> {
  let expanded = expand_and_combine(expr);
  // Constant term: substitute var=0
  let constant =
    crate::syntax::substitute_variable(&expanded, var, &Expr::Integer(0));
  let constant =
    crate::evaluator::evaluate_expr_to_expr(&constant).unwrap_or(constant);
  // Value at var=1: substitute var=1
  let at_one =
    crate::syntax::substitute_variable(&expanded, var, &Expr::Integer(1));
  let at_one =
    crate::evaluator::evaluate_expr_to_expr(&at_one).unwrap_or(at_one);
  // linear_coeff = at_one - constant
  let coeff = Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(at_one),
    right: Box::new(constant.clone()),
  };
  let coeff = crate::evaluator::evaluate_expr_to_expr(&coeff).unwrap_or(coeff);
  // Check that the coefficient is not zero (otherwise not linear)
  if matches!(&coeff, Expr::Integer(0)) {
    return None;
  }
  // Verify it's actually linear (degree 1): check that at var=2, value = 2*coeff + constant
  let at_two =
    crate::syntax::substitute_variable(&expanded, var, &Expr::Integer(2));
  let at_two =
    crate::evaluator::evaluate_expr_to_expr(&at_two).unwrap_or(at_two);
  // expected = 2*coeff + constant
  let expected = Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left: Box::new(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(2)),
      right: Box::new(coeff.clone()),
    }),
    right: Box::new(constant.clone()),
  };
  let expected =
    crate::evaluator::evaluate_expr_to_expr(&expected).unwrap_or(expected);
  let diff = Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(at_two),
    right: Box::new(expected),
  };
  let diff = crate::evaluator::evaluate_expr_to_expr(&diff).unwrap_or(diff);
  if !matches!(&diff, Expr::Integer(0)) {
    return None; // Not linear
  }
  Some((coeff, constant))
}

/// Extract leading sign from a Times expression.
/// Returns (sign, abs_expr) where sign is 1 or -1.
fn extract_leading_sign(expr: &Expr) -> (i128, Expr) {
  match expr {
    Expr::Integer(n) if *n < 0 => (-1, Expr::Integer(-n)),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => (-1, *operand.clone()),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let (lsign, labs) = extract_leading_sign(left);
      if lsign < 0 {
        let rebuilt =
          crate::functions::math_ast::times_ast(&[labs, *right.clone()])
            .unwrap_or(*right.clone());
        (-1, rebuilt)
      } else {
        (1, expr.clone())
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Times" && !args.is_empty() =>
    {
      let (lsign, labs) = extract_leading_sign(&args[0]);
      if lsign < 0 {
        let mut new_args = vec![labs];
        new_args.extend_from_slice(&args[1..]);
        let rebuilt = crate::functions::math_ast::times_ast(&new_args)
          .unwrap_or(expr.clone());
        (-1, rebuilt)
      } else {
        (1, expr.clone())
      }
    }
    _ => (1, expr.clone()),
  }
}

/// Symbolic partial fraction decomposition for multivariate expressions.
/// Factor the denominator, find linear factors in var, apply cover-up method.
fn apart_symbolic(
  expr: &Expr,
  num: &Expr,
  den: &Expr,
  var: &str,
) -> Result<Expr, InterpreterError> {
  // Factor the denominator
  let den_factored = factor_ast(&[den.clone()])?;
  // Collect factors
  let mut factors = Vec::new();
  flatten_product_factors(&den_factored, &mut factors);

  // Separate linear (in var) factors from non-linear/constant factors
  let mut linear_factors: Vec<Expr> = Vec::new(); // the full factor expression
  let mut linear_roots: Vec<Expr> = Vec::new(); // root = -constant/coeff
  let mut other_factor = Expr::Integer(1);

  for f in &factors {
    if !crate::functions::polynomial_ast::contains_var(f, var) {
      // Constant factor (doesn't contain var)
      other_factor = crate::functions::math_ast::times_ast(&[
        other_factor.clone(),
        f.clone(),
      ])
      .unwrap_or(other_factor);
      continue;
    }
    if let Some((coeff, constant)) = extract_linear_coeffs(f, var) {
      // Root: var = -constant/coeff
      let neg_const = Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(constant),
      };
      let root = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(neg_const),
        right: Box::new(coeff),
      };
      let root = crate::evaluator::evaluate_expr_to_expr(&root).unwrap_or(root);
      linear_factors.push(f.clone());
      linear_roots.push(root);
    } else {
      // Non-linear factor containing var — can't decompose
      return Ok(expr.clone());
    }
  }

  if linear_factors.len() < 2 {
    return Ok(expr.clone());
  }

  // Apply cover-up method for partial fractions
  let n = linear_roots.len();
  let mut result_terms = Vec::new();

  for i in 0..n {
    // Evaluate numerator at root_i
    let num_val =
      crate::syntax::substitute_variable(num, var, &linear_roots[i]);
    let num_val =
      crate::evaluator::evaluate_expr_to_expr(&num_val).unwrap_or(num_val);

    // Compute scalar = other_factor * product_{j!=i} factor_j(r_i)
    let mut scalar_parts: Vec<Expr> = vec![other_factor.clone()];
    for j in 0..n {
      if i != j {
        let fj_at_ri = crate::syntax::substitute_variable(
          &linear_factors[j],
          var,
          &linear_roots[i],
        );
        let fj_at_ri = crate::evaluator::evaluate_expr_to_expr(&fj_at_ri)
          .unwrap_or(fj_at_ri);
        scalar_parts.push(fj_at_ri);
      }
    }
    let scalar = crate::functions::math_ast::times_ast(&scalar_parts)
      .unwrap_or(Expr::Integer(1));

    // Extract sign from scalar: if leading coefficient is negative, flip sign
    let (sign, abs_scalar) = extract_leading_sign(&scalar);

    // Build denominator = abs_scalar * factor_i (always positive scalar)
    let mut denom_factors = Vec::new();
    flatten_product_factors(&abs_scalar, &mut denom_factors);
    denom_factors.push(linear_factors[i].clone());
    let full_denom = crate::functions::math_ast::times_ast(&denom_factors)
      .unwrap_or(Expr::Integer(1));

    // Build positive fraction = num_val / full_denom
    let frac = Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(num_val),
      right: Box::new(full_denom),
    };
    let frac = crate::evaluator::evaluate_expr_to_expr(&frac).unwrap_or(frac);

    // Apply sign: negative terms use UnaryOp::Minus for proper plus_ast handling
    if sign < 0 {
      result_terms.push(Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(frac),
      });
    } else {
      result_terms.push(frac);
    }
  }

  if result_terms.is_empty() {
    return Ok(expr.clone());
  }

  // Build sum: positive terms first, then negative terms (as subtractions)
  let mut positive: Vec<Expr> = Vec::new();
  let mut negative: Vec<Expr> = Vec::new(); // stored as absolute values
  for t in &result_terms {
    match t {
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => negative.push(*operand.clone()),
      _ => positive.push(t.clone()),
    }
  }
  // Build: pos1 + pos2 + ... - neg1 - neg2 - ...
  let mut result = if positive.is_empty() {
    // All negative: start with -neg1
    if negative.is_empty() {
      return Ok(expr.clone());
    }
    let first = negative.remove(0);
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(first),
    }
  } else {
    positive.remove(0)
  };
  for p in &positive {
    result = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(result),
      right: Box::new(p.clone()),
    };
  }
  for n in &negative {
    result = Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(result),
      right: Box::new(n.clone()),
    };
  }
  Ok(result)
}
