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
              // Power[fraction, n] → split base into num/den
              let (base_num, base_den) = extract_num_den(&pargs[0]);
              if !matches!(&base_den, Expr::Integer(1)) {
                num_factors.push(Expr::FunctionCall {
                  name: "Power".to_string(),
                  args: vec![base_num, pargs[1].clone()],
                });
                den_factors.push(Expr::FunctionCall {
                  name: "Power".to_string(),
                  args: vec![base_den, pargs[1].clone()],
                });
              } else {
                num_factors.push(arg.clone());
              }
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
              // Power[fraction, n] → split base into num/den
              let (base_num, base_den) = extract_num_den(left);
              if !matches!(&base_den, Expr::Integer(1)) {
                num_factors.push(Expr::BinaryOp {
                  op: BinaryOperator::Power,
                  left: Box::new(base_num),
                  right: right.clone(),
                });
                den_factors.push(Expr::BinaryOp {
                  op: BinaryOperator::Power,
                  left: Box::new(base_den),
                  right: right.clone(),
                });
              } else {
                num_factors.push(arg.clone());
              }
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
          // Rational[n,d] inside Times: split numerator n and denominator d
          Expr::FunctionCall {
            name: rname,
            args: rargs,
          } if rname == "Rational" && rargs.len() == 2 => {
            // Denominator is always positive after make_rational normalisation
            num_factors.push(rargs[0].clone());
            den_factors.push(rargs[1].clone());
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
    // BinaryOp::Power — handle negative exponents and Power[fraction, n]
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      if let Some(neg_exp) = get_negative_integer(right) {
        if neg_exp == 1 {
          (Expr::Integer(1), *left.clone())
        } else {
          (
            Expr::Integer(1),
            Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: left.clone(),
              right: Box::new(Expr::Integer(neg_exp as i128)),
            },
          )
        }
      } else {
        // Power[num/den, n] → (num^n, den^n)
        let (base_num, base_den) = extract_num_den(left);
        if !matches!(&base_den, Expr::Integer(1)) {
          let num = Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(base_num),
            right: right.clone(),
          };
          let den = Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(base_den),
            right: right.clone(),
          };
          (num, den)
        } else {
          (expr.clone(), Expr::Integer(1))
        }
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

  // Compute the common denominator (LCM of all denominators)
  // Decompose each denominator into base^exp pairs and take max exp for each base
  let mut base_exp_map: Vec<(String, Expr, i128)> = Vec::new(); // (key, base, max_exp)
  for (_, den) in &fractions {
    if matches!(den, Expr::Integer(1)) {
      continue;
    }
    let den_factors = extract_den_factors(den);
    for (base, exp) in &den_factors {
      let key = expr_to_string(base);
      if let Some(entry) = base_exp_map.iter_mut().find(|(k, _, _)| *k == key) {
        entry.2 = entry.2.max(*exp); // Take max exponent (LCM)
      } else {
        base_exp_map.push((key, base.clone(), *exp));
      }
    }
  }

  if base_exp_map.is_empty() {
    // No fractions to combine
    return expr.clone();
  }

  // Build numerator: for each term, multiply by (common_den / den_i)
  let mut new_num_terms = Vec::new();
  for (num, den) in &fractions {
    let missing_factor = compute_missing_factor(den, &base_exp_map);
    new_num_terms.push(multiply_exprs(num, &missing_factor));
  }

  let combined_num = if new_num_terms.len() == 1 {
    expand_and_combine(&new_num_terms.remove(0))
  } else {
    expand_and_combine(&build_sum(new_num_terms))
  };
  // Keep denominator in factored form (Wolfram behavior),
  // but canonicalize each individual factor and sort them
  let combined_den = {
    let mut canonical_dens: Vec<Expr> = base_exp_map
      .iter()
      .map(|(_, base, exp)| {
        if *exp == 1 {
          expand_and_combine(base)
        } else {
          expand_and_combine(&Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(base.clone()),
            right: Box::new(Expr::Integer(*exp)),
          })
        }
      })
      .collect();
    canonical_dens.sort_by_key(expr_to_string);
    if canonical_dens.len() == 1 {
      canonical_dens.remove(0)
    } else {
      build_product(canonical_dens)
    }
  };

  if matches!(&combined_num, Expr::Integer(0)) {
    Expr::Integer(0)
  } else if matches!(&combined_den, Expr::Integer(1)) {
    combined_num
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(combined_num),
      right: Box::new(combined_den),
    }
  }
}

/// Extract base and exponent from a denominator expression.
/// E.g. `(2*a)^2` → [((2*a), 2)], `2*a` → [(2*a, 1)]
/// Keeps products as single bases to allow matching: e.g. `2*a` matches base of `(2*a)^2`.
fn extract_den_factors(den: &Expr) -> Vec<(Expr, i128)> {
  match den {
    Expr::Integer(1) => vec![],
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      if let Expr::Integer(n) = right.as_ref() {
        vec![(*left.clone(), *n)]
      } else {
        vec![(den.clone(), 1)]
      }
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if let Expr::Integer(n) = &args[1] {
        vec![(args[0].clone(), *n)]
      } else {
        vec![(den.clone(), 1)]
      }
    }
    _ => vec![(den.clone(), 1)],
  }
}

/// Compute the "missing factor" needed to bring a fraction's denominator up to the common
/// denominator. For each base in the LCM, compute base^(lcm_exp - den_exp).
fn compute_missing_factor(
  den: &Expr,
  base_exp_map: &[(String, Expr, i128)],
) -> Expr {
  let den_factors = extract_den_factors(den);
  let mut den_map: Vec<(String, i128)> = Vec::new();
  for (base, exp) in &den_factors {
    let key = expr_to_string(base);
    if let Some(entry) = den_map.iter_mut().find(|(k, _)| *k == key) {
      entry.1 += exp;
    } else {
      den_map.push((key, *exp));
    }
  }

  let mut missing_factors: Vec<Expr> = Vec::new();
  for (key, base, lcm_exp) in base_exp_map {
    let den_exp = den_map
      .iter()
      .find(|(k, _)| k == key)
      .map(|(_, e)| *e)
      .unwrap_or(0);
    let diff = lcm_exp - den_exp;
    if diff > 0 {
      if diff == 1 {
        missing_factors.push(base.clone());
      } else {
        missing_factors.push(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(base.clone()),
          right: Box::new(Expr::Integer(diff)),
        });
      }
    }
  }

  if missing_factors.is_empty() {
    Expr::Integer(1)
  } else {
    build_product(missing_factors)
  }
}
