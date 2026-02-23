#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr};

/// Cyclotomic[n, x] - The n-th cyclotomic polynomial evaluated at x
pub fn cyclotomic_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Cyclotomic expects exactly 2 arguments".into(),
    ));
  }

  let n = match &args[0] {
    Expr::Integer(n) => {
      if *n < 0 {
        return Err(InterpreterError::EvaluationError(
          "Cyclotomic: first argument must be a non-negative integer".into(),
        ));
      }
      *n as usize
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Cyclotomic".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Φ_0(x) = 1
  if n == 0 {
    return Ok(Expr::Integer(1));
  }

  // Compute cyclotomic polynomial coefficients
  let coeffs = cyclotomic_coeffs(n);

  // Build the polynomial expression in x
  let x = &args[1];
  build_polynomial_expr(&coeffs, x)
}

/// Compute the coefficients of the n-th cyclotomic polynomial.
/// Returns coefficients from degree 0 to degree phi(n).
fn cyclotomic_coeffs(n: usize) -> Vec<i128> {
  if n == 1 {
    // Φ_1(x) = x - 1
    return vec![-1, 1];
  }

  // Start with x^n - 1
  let mut result = vec![0i128; n + 1];
  result[0] = -1;
  result[n] = 1;

  // Divide by Φ_d(x) for all proper divisors d of n
  let divisors = proper_divisors(n);
  for d in divisors {
    let phi_d = cyclotomic_coeffs(d);
    result = poly_div_exact(&result, &phi_d);
  }

  result
}

/// Find all proper divisors of n (excluding n itself, including 1)
fn proper_divisors(n: usize) -> Vec<usize> {
  let mut divs = Vec::new();
  for d in 1..n {
    if n.is_multiple_of(d) {
      divs.push(d);
    }
  }
  divs
}

/// Exact polynomial division (quotient only, assumes exact division)
fn poly_div_exact(dividend: &[i128], divisor: &[i128]) -> Vec<i128> {
  let n = dividend.len();
  let m = divisor.len();
  if m > n || m == 0 {
    return dividend.to_vec();
  }

  let mut rem = dividend.to_vec();
  let lead = divisor[m - 1];
  let quot_len = n - m + 1;
  let mut quot = vec![0i128; quot_len];

  for i in (0..quot_len).rev() {
    let coeff = rem[i + m - 1] / lead;
    quot[i] = coeff;
    for j in 0..m {
      rem[i + j] -= coeff * divisor[j];
    }
  }

  // Trim trailing zeros
  while quot.len() > 1 && *quot.last().unwrap() == 0 {
    quot.pop();
  }

  quot
}

/// Build a polynomial expression from integer coefficients and a variable
fn build_polynomial_expr(
  coeffs: &[i128],
  x: &Expr,
) -> Result<Expr, InterpreterError> {
  let mut terms: Vec<Expr> = Vec::new();

  for (i, &c) in coeffs.iter().enumerate() {
    if c == 0 {
      continue;
    }
    let term = if i == 0 {
      Expr::Integer(c)
    } else if i == 1 {
      if c == 1 {
        x.clone()
      } else if c == -1 {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(-1)),
          right: Box::new(x.clone()),
        }
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(c)),
          right: Box::new(x.clone()),
        }
      }
    } else {
      let x_pow = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(x.clone()),
        right: Box::new(Expr::Integer(i as i128)),
      };
      if c == 1 {
        x_pow
      } else if c == -1 {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(-1)),
          right: Box::new(x_pow),
        }
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(c)),
          right: Box::new(x_pow),
        }
      }
    };
    terms.push(term);
  }

  if terms.is_empty() {
    return Ok(Expr::Integer(0));
  }

  // Evaluate to get proper canonical form
  let mut result = terms.remove(0);
  for t in terms {
    result = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(result),
      right: Box::new(t),
    };
  }

  // Evaluate to simplify and canonicalize
  crate::evaluator::evaluate_expr_to_expr(&result)
}
