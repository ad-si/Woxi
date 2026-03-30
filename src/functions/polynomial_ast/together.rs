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
pub fn extract_num_den(expr: &Expr) -> (Expr, Expr) {
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
    // Power[base, -n] => 1/base^n (FunctionCall form) — handles integer and rational exponents
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if let Some(pos_exp) = get_negative_exponent(&args[1]) {
        if matches!(&pos_exp, Expr::Integer(1)) {
          (Expr::Integer(1), args[0].clone())
        } else {
          (
            Expr::Integer(1),
            Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![args[0].clone(), pos_exp],
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
            if let Some(pos_exp) = get_negative_exponent(&pargs[1]) {
              if matches!(&pos_exp, Expr::Integer(1)) {
                den_factors.push(pargs[0].clone());
              } else {
                den_factors.push(Expr::FunctionCall {
                  name: "Power".to_string(),
                  args: vec![pargs[0].clone(), pos_exp],
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
            if let Some(pos_exp) = get_negative_exponent(right) {
              if matches!(&pos_exp, Expr::Integer(1)) {
                den_factors.push(*left.clone());
              } else {
                den_factors.push(Expr::FunctionCall {
                  name: "Power".to_string(),
                  args: vec![*left.clone(), pos_exp],
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
      if let Some(pos_exp) = get_negative_exponent(right) {
        if matches!(&pos_exp, Expr::Integer(1)) {
          (Expr::Integer(1), *left.clone())
        } else {
          (
            Expr::Integer(1),
            Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: left.clone(),
              right: Box::new(pos_exp),
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

/// Check if an expression is a negative exponent (integer or rational)
/// and return the negated (positive) exponent.
pub fn get_negative_exponent(expr: &Expr) -> Option<Expr> {
  // Try integer first
  if let Some(neg) = get_negative_integer(expr) {
    return Some(Expr::Integer(neg as i128));
  }
  // Try Rational[-n, d]
  if let Expr::FunctionCall { name, args } = expr
    && name == "Rational"
    && args.len() == 2
    && let Expr::Integer(n) = &args[0]
    && *n < 0
  {
    return Some(Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(-*n), args[1].clone()],
    });
  }
  // Try Times[-1, Rational[p, q]] → negative rational
  if let Expr::FunctionCall { name, args } = expr
    && name == "Times"
    && args.len() == 2
    && matches!(&args[0], Expr::Integer(-1))
    && let Expr::FunctionCall { name: rn, args: ra } = &args[1]
    && rn == "Rational"
    && ra.len() == 2
    && let Expr::Integer(n) = &ra[0]
    && *n > 0
  {
    return Some(Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(*n), ra[1].clone()],
    });
  }
  None
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
  let mut base_exp_map: Vec<(String, Expr, RatExp)> = Vec::new();
  for (_, den) in &fractions {
    if matches!(den, Expr::Integer(1)) {
      continue;
    }
    let den_factors = extract_den_factors(den);
    for (base, exp) in &den_factors {
      let key = expr_to_string(base);
      if let Some(entry) = base_exp_map.iter_mut().find(|(k, _, _)| *k == key) {
        entry.2 = rat_max(entry.2, *exp); // Take max exponent (LCM)
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
        if *exp == (1, 1) {
          expand_and_combine(base)
        } else {
          expand_and_combine(&Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(base.clone()),
            right: Box::new(rat_exp_to_expr(*exp)),
          })
        }
      })
      .collect();
    crate::functions::math_ast::sort_symbolic_factors(&mut canonical_dens);
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
    // Try to cancel common monomial factors between numerator and denominator
    // without re-expanding the denominator (preserves factored form).
    let (simplified_num, simplified_den) =
      cancel_common_monomial_factors(&combined_num, &combined_den);
    if matches!(&simplified_den, Expr::Integer(1)) {
      simplified_num
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(simplified_num),
        right: Box::new(simplified_den),
      }
    }
  }
}

/// A rational exponent represented as (numerator, denominator) with denominator > 0.
type RatExp = (i128, i128);

fn rat_exp_from_expr(exp: &Expr) -> RatExp {
  match exp {
    Expr::Integer(n) => (*n, 1),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        (*n, *d)
      } else {
        (1, 1)
      }
    }
    _ => (1, 1),
  }
}

fn rat_exp_to_expr((n, d): RatExp) -> Expr {
  if d == 1 {
    Expr::Integer(n)
  } else {
    Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(n), Expr::Integer(d)],
    }
  }
}

/// Compare two rational exponents: returns a > b
fn rat_gt((an, ad): RatExp, (bn, bd): RatExp) -> bool {
  an * bd > bn * ad
}

/// Compute max of two rational exponents
fn rat_max(a: RatExp, b: RatExp) -> RatExp {
  if rat_gt(a, b) { a } else { b }
}

/// Subtract two rational exponents: a - b
fn rat_sub((an, ad): RatExp, (bn, bd): RatExp) -> RatExp {
  let n = an * bd - bn * ad;
  let d = ad * bd;
  let g = crate::functions::math_ast::gcd(n, d);
  (n / g, d / g)
}

/// Extract base and exponent from a denominator expression.
/// Returns (base, exponent) pairs with rational exponents.
fn extract_den_factors(den: &Expr) -> Vec<(Expr, RatExp)> {
  match den {
    Expr::Integer(1) => vec![],
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      vec![(*left.clone(), rat_exp_from_expr(right))]
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      vec![(args[0].clone(), rat_exp_from_expr(&args[1]))]
    }
    // Times[a, b, ...] in denominator — split into factors
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      let mut result = Vec::new();
      for arg in args {
        result.extend(extract_den_factors(arg));
      }
      result
    }
    _ => vec![(den.clone(), (1, 1))],
  }
}

/// Compute the "missing factor" needed to bring a fraction's denominator up to the common
/// denominator. For each base in the LCM, compute base^(lcm_exp - den_exp).
fn compute_missing_factor(
  den: &Expr,
  base_exp_map: &[(String, Expr, RatExp)],
) -> Expr {
  let den_factors = extract_den_factors(den);
  let mut den_map: Vec<(String, RatExp)> = Vec::new();
  for (base, exp) in &den_factors {
    let key = expr_to_string(base);
    if let Some(entry) = den_map.iter_mut().find(|(k, _)| *k == key) {
      // Add exponents for same base
      entry.1 = (entry.1.0 * exp.1 + exp.0 * entry.1.1, entry.1.1 * exp.1);
      let g = crate::functions::math_ast::gcd(entry.1.0, entry.1.1);
      entry.1 = (entry.1.0 / g, entry.1.1 / g);
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
      .unwrap_or((0, 1));
    let diff = rat_sub(*lcm_exp, den_exp);
    if rat_gt(diff, (0, 1)) {
      if diff == (1, 1) {
        missing_factors.push(base.clone());
      } else {
        missing_factors.push(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(base.clone()),
          right: Box::new(rat_exp_to_expr(diff)),
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

/// Cancel common monomial factors between a numerator (sum of terms) and a denominator (product).
/// E.g. (a^2*x + a*x^2) / a^2 → (a*x + x^2) / a
/// Does not expand the denominator, preserving its factored form.
fn cancel_common_monomial_factors(num: &Expr, den: &Expr) -> (Expr, Expr) {
  let terms = collect_additive_terms(num);
  if terms.len() < 2 {
    return (num.clone(), den.clone());
  }

  // For each term, extract base→exp map of its multiplicative factors (ignoring integers)
  fn term_base_exp(term: &Expr) -> Vec<(String, Expr, i128)> {
    let factors = flatten_times_args(&[term.clone()]);
    let mut map: Vec<(String, Expr, i128)> = Vec::new();
    for f in &factors {
      let (base_str, base, exp) = match f {
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left,
          right,
        } => {
          if let Expr::Integer(n) = right.as_ref() {
            if *n > 0 {
              (expr_to_string(left), *left.clone(), *n)
            } else {
              continue;
            }
          } else {
            continue;
          }
        }
        Expr::FunctionCall { name, args }
          if name == "Power" && args.len() == 2 =>
        {
          if let Expr::Integer(n) = &args[1] {
            if *n > 0 {
              (expr_to_string(&args[0]), args[0].clone(), *n)
            } else {
              continue;
            }
          } else {
            continue;
          }
        }
        Expr::Integer(_) => continue,
        _ => (expr_to_string(f), f.clone(), 1),
      };
      if let Some(entry) = map.iter_mut().find(|(k, _, _)| *k == base_str) {
        entry.2 += exp;
      } else {
        map.push((base_str, base, exp));
      }
    }
    map
  }

  // Find common base^exp across all terms (min exponent for each base)
  let mut common = term_base_exp(&terms[0]);
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

  if common.is_empty() {
    return (num.clone(), den.clone());
  }

  // Check which common factors also appear in the denominator and can be cancelled
  let den_factors = flatten_times_args(&[den.clone()]);
  let mut den_map: Vec<(String, Expr, i128)> = Vec::new();
  for f in &den_factors {
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
    if let Some(entry) = den_map.iter_mut().find(|(k, _, _)| *k == base_str) {
      entry.2 += exp;
    } else {
      den_map.push((base_str, base, exp));
    }
  }

  // Determine how much of each common factor can be cancelled with the denominator
  let mut cancel_map: Vec<(String, i128)> = Vec::new();
  for (key, _, num_exp) in &common {
    if let Some((_, _, den_exp)) = den_map.iter().find(|(k, _, _)| k == key) {
      let cancel_exp = (*num_exp).min(*den_exp);
      if cancel_exp > 0 {
        cancel_map.push((key.clone(), cancel_exp));
      }
    }
  }

  if cancel_map.is_empty() {
    return (num.clone(), den.clone());
  }

  // Divide each numerator term by the cancelled factors
  let mut new_terms = Vec::new();
  for term in &terms {
    let mut t_factors: Vec<Expr> = flatten_times_args(&[term.clone()]);
    for (cancel_key, cancel_exp) in &cancel_map {
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
              new_factors.push(f);
              continue;
            }
          }
          Expr::FunctionCall { name, args }
            if name == "Power" && args.len() == 2 =>
          {
            if let Expr::Integer(n) = &args[1] {
              (expr_to_string(&args[0]), args[0].clone(), *n)
            } else {
              new_factors.push(f);
              continue;
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
          // new_exp == 0: factor removed
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

  // Build new numerator
  let new_num = if new_terms.len() == 1 {
    expand_and_combine(&new_terms[0])
  } else {
    expand_and_combine(&build_sum(new_terms))
  };

  // Build new denominator: reduce exponents of cancelled factors
  let mut new_den_factors: Vec<Expr> = Vec::new();
  // Keep integer factors
  for f in &den_factors {
    if let Expr::Integer(n) = f {
      new_den_factors.push(Expr::Integer(*n));
    }
  }
  let mut den_map_remaining = den_map.clone();
  for (cancel_key, cancel_exp) in &cancel_map {
    if let Some(entry) = den_map_remaining
      .iter_mut()
      .find(|(k, _, _)| k == cancel_key)
    {
      entry.2 -= cancel_exp;
    }
  }
  for (_, base, exp) in &den_map_remaining {
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
  // Sort non-numeric factors to match Wolfram canonical order
  let numeric_end = new_den_factors
    .iter()
    .position(|f| !matches!(f, Expr::Integer(_)))
    .unwrap_or(new_den_factors.len());
  crate::functions::math_ast::sort_symbolic_factors(
    &mut new_den_factors[numeric_end..],
  );
  let new_den = if new_den_factors.is_empty() {
    Expr::Integer(1)
  } else {
    build_product(new_den_factors)
  };

  (new_num, new_den)
}
