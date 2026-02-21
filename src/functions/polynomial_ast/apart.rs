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
  // Extract numerator and denominator
  let (num, den) = match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => (*left.clone(), *right.clone()),
    _ => {
      return Ok(expr.clone());
    }
  };

  let num_expanded = expand_and_combine(&num);
  let den_expanded = expand_and_combine(&den);

  let num_coeffs = match extract_poly_coeffs(&num_expanded, var) {
    Some(c) => c,
    None => return Ok(expr.clone()),
  };
  let den_coeffs = match extract_poly_coeffs(&den_expanded, var) {
    Some(c) => c,
    None => return Ok(expr.clone()),
  };

  // If numerator degree >= denominator degree, do polynomial division first
  if num_coeffs.len() >= den_coeffs.len() {
    if let Some(quot_coeffs) = poly_exact_divide(&num_coeffs, &den_coeffs) {
      // Perfectly divisible
      return Ok(coeffs_to_expr(&quot_coeffs, var));
    }
    // Polynomial long division with remainder
    let (quotient, remainder) = poly_long_divide(&num_coeffs, &den_coeffs);
    if remainder.iter().all(|&c| c == 0) {
      return Ok(coeffs_to_expr(&quotient, var));
    }
    // result = quotient + Apart[remainder/den]
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

  apart_proper_fraction(expr, var)
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
