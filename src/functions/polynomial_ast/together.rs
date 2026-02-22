#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator, expr_to_string};

// ─── Together ───────────────────────────────────────────────────────

/// Together[expr] - Combines fractions over a common denominator
/// Threads over List.
pub fn together_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Together expects exactly 1 argument".into(),
    ));
  }
  // Thread over List
  if let Expr::List(items) = &args[0] {
    let results: Vec<Expr> = items.iter().map(together_expr).collect();
    return Ok(Expr::List(results));
  }
  Ok(together_expr(&args[0]))
}

/// Extract numerator and denominator from an expression.
/// Handles BinaryOp::Divide, Rational, Power[..., -1], and
/// Times[..., Power[..., -1]] forms.
pub(super) fn extract_num_den(expr: &Expr) -> (Expr, Expr) {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => (*left.clone(), *right.clone()),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      (args[0].clone(), args[1].clone())
    }
    // Power[base, -n] => 1/base^n (FunctionCall form)
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if let Some(neg_exp) = get_negative_integer(&args[1]) {
        if neg_exp == 1 {
          (Expr::Integer(1), args[0].clone())
        } else {
          (
            Expr::Integer(1),
            Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![args[0].clone(), Expr::Integer(neg_exp as i128)],
            },
          )
        }
      } else {
        (expr.clone(), Expr::Integer(1))
      }
    }
    // Times[factors...] — split into numerator factors and denominator factors
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      // Flatten nested Times first
      let flat_args = flatten_times_args(args);
      let mut num_factors = Vec::new();
      let mut den_factors = Vec::new();
      for arg in &flat_args {
        match arg {
          Expr::FunctionCall {
            name: pname,
            args: pargs,
          } if pname == "Power" && pargs.len() == 2 => {
            if let Some(neg_exp) = get_negative_integer(&pargs[1]) {
              if neg_exp == 1 {
                den_factors.push(pargs[0].clone());
              } else {
                den_factors.push(Expr::FunctionCall {
                  name: "Power".to_string(),
                  args: vec![pargs[0].clone(), Expr::Integer(neg_exp as i128)],
                });
              }
            } else {
              num_factors.push(arg.clone());
            }
          }
          Expr::BinaryOp {
            op: BinaryOperator::Power,
            left,
            right,
          } => {
            if let Some(neg_exp) = get_negative_integer(right) {
              if neg_exp == 1 {
                den_factors.push(*left.clone());
              } else {
                den_factors.push(Expr::FunctionCall {
                  name: "Power".to_string(),
                  args: vec![*left.clone(), Expr::Integer(neg_exp as i128)],
                });
              }
            } else {
              num_factors.push(arg.clone());
            }
          }
          // BinaryOp::Divide inside Times: split into num/den
          Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left,
            right,
          } => {
            num_factors.push(*left.clone());
            den_factors.push(*right.clone());
          }
          _ => num_factors.push(arg.clone()),
        }
      }
      if den_factors.is_empty() {
        (expr.clone(), Expr::Integer(1))
      } else {
        let num = build_product(num_factors);
        let den = build_product(den_factors);
        (num, den)
      }
    }
    // BinaryOp::Times — split into numerator and denominator factors
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      // Flatten into a Times FunctionCall and recurse
      let flat = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![*left.clone(), *right.clone()],
      };
      extract_num_den(&flat)
    }
    // UnaryOp::Minus — negate the numerator
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let (num, den) = extract_num_den(operand);
      (negate_expr(&num), den)
    }
    _ => (expr.clone(), Expr::Integer(1)),
  }
}

/// Flatten nested Times args: Times[a, Times[b, c]] → [a, b, c]
pub fn flatten_times_args(args: &[Expr]) -> Vec<Expr> {
  let mut flat = Vec::new();
  for arg in args {
    match arg {
      Expr::FunctionCall { name, args: inner } if name == "Times" => {
        flat.extend(flatten_times_args(inner));
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        flat.extend(flatten_times_args(&[*left.clone(), *right.clone()]));
      }
      _ => flat.push(arg.clone()),
    }
  }
  flat
}

/// Check if an expression is a negative integer and return its absolute value
pub fn get_negative_integer(expr: &Expr) -> Option<i64> {
  match expr {
    Expr::Integer(n) if *n < 0 => Some((-*n) as i64),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => match operand.as_ref() {
      Expr::Integer(n) if *n > 0 => Some(*n as i64),
      _ => None,
    },
    _ => None,
  }
}

/// Negate an expression
pub(super) fn negate_expr(expr: &Expr) -> Expr {
  match expr {
    Expr::Integer(n) => Expr::Integer(-n),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => *operand.clone(),
    _ => Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(expr.clone()),
    },
  }
}

pub fn together_expr(expr: &Expr) -> Expr {
  // Collect additive terms and put them over a common denominator
  let terms = collect_additive_terms(expr);
  if terms.len() <= 1 {
    return expr.clone();
  }

  // Extract numerator and denominator for each term
  let mut fractions: Vec<(Expr, Expr)> = Vec::new();
  for term in &terms {
    fractions.push(extract_num_den(term));
  }

  // Compute the common denominator (product of all denominators, simplified)
  let mut common_den = Expr::Integer(1);
  let mut unique_dens: Vec<Expr> = Vec::new();
  for (_, den) in &fractions {
    if !matches!(den, Expr::Integer(1)) {
      let den_str = expr_to_string(den);
      if !unique_dens.iter().any(|d| expr_to_string(d) == den_str) {
        unique_dens.push(den.clone());
        common_den = multiply_exprs(&common_den, den);
      }
    }
  }

  if matches!(&common_den, Expr::Integer(1)) {
    // No fractions to combine
    return expr.clone();
  }

  // Build numerator: sum of (num_i * common_den / den_i)
  let mut new_num_terms = Vec::new();
  for (num, den) in &fractions {
    if matches!(den, Expr::Integer(1)) {
      // Multiply by full common_den
      new_num_terms.push(multiply_exprs(num, &common_den));
    } else {
      // Multiply by common_den / den
      let mut factor = Expr::Integer(1);
      for ud in &unique_dens {
        let ud_str = expr_to_string(ud);
        let den_str = expr_to_string(den);
        if ud_str != den_str {
          factor = multiply_exprs(&factor, ud);
        }
      }
      new_num_terms.push(multiply_exprs(num, &factor));
    }
  }

  let combined_num = if new_num_terms.len() == 1 {
    expand_and_combine(&new_num_terms.remove(0))
  } else {
    expand_and_combine(&build_sum(new_num_terms))
  };
  // Keep denominator in factored form (Wolfram behavior),
  // but canonicalize each individual factor and sort them
  let combined_den = if unique_dens.len() == 1 {
    expand_and_combine(&unique_dens[0])
  } else {
    let mut canonical_dens: Vec<Expr> =
      unique_dens.iter().map(expand_and_combine).collect();
    canonical_dens.sort_by_key(expr_to_string);
    build_product(canonical_dens)
  };

  if matches!(&combined_den, Expr::Integer(1)) {
    combined_num
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(combined_num),
      right: Box::new(combined_den),
    }
  }
}
