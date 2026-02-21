#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator, expr_to_string};

use crate::functions::calculus_ast::simplify;

// ─── Expand ─────────────────────────────────────────────────────────

/// Expand[expr] - Expands products and positive integer powers
pub fn expand_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Expand expects exactly 1 argument".into(),
    ));
  }
  // Thread over Lists
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| expand_ast(&[item.clone()]))
      .collect();
    return Ok(Expr::List(results?));
  }
  // Thread over Rules
  if let Expr::Rule {
    pattern,
    replacement,
  } = &args[0]
  {
    let expanded_pattern = expand_and_combine(pattern);
    let expanded_replacement = expand_and_combine(replacement);
    return Ok(Expr::Rule {
      pattern: Box::new(expanded_pattern),
      replacement: Box::new(expanded_replacement),
    });
  }
  Ok(expand_and_combine(&args[0]))
}

/// Expand an expression and combine like terms.
pub fn expand_and_combine(expr: &Expr) -> Expr {
  let expanded = expand_expr(expr);
  let terms = collect_additive_terms(&expanded);
  combine_and_build(terms)
}

/// Recursively expand an expression.
pub fn expand_expr(expr: &Expr) -> Expr {
  match expr {
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::String(_)
    | Expr::Constant(_)
    | Expr::Identifier(_)
    | Expr::Slot(_) => expr.clone(),

    Expr::BinaryOp { op, left, right } => {
      let left_exp = expand_expr(left);
      let right_exp = expand_expr(right);
      match op {
        BinaryOperator::Plus => Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(left_exp),
          right: Box::new(right_exp),
        },
        BinaryOperator::Minus => Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(left_exp),
          right: Box::new(right_exp),
        },
        BinaryOperator::Times => distribute_product(&left_exp, &right_exp),
        BinaryOperator::Power => {
          // (sum)^n where n is positive integer
          if let Expr::Integer(n) = &right_exp
            && *n >= 2
            && is_sum(&left_exp)
          {
            return expand_power(&left_exp, *n);
          }
          // Try to simplify Power (e.g. I^2 → -1, Sqrt[x]^2 → x)
          if let Ok(simplified) = crate::functions::math_ast::power_ast(&[
            left_exp.clone(),
            right_exp.clone(),
          ]) {
            // Only use simplified result if it actually simplified
            if !matches!(
              &simplified,
              Expr::BinaryOp {
                op: BinaryOperator::Power,
                ..
              }
            ) {
              return simplified;
            }
          }
          Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(left_exp),
            right: Box::new(right_exp),
          }
        }
        _ => Expr::BinaryOp {
          op: *op,
          left: Box::new(left_exp),
          right: Box::new(right_exp),
        },
      }
    }

    Expr::UnaryOp { op, operand } => {
      let operand_exp = expand_expr(operand);
      match op {
        UnaryOperator::Minus => {
          // Distribute minus over sums
          let terms = collect_additive_terms(&operand_exp);
          let negated: Vec<Expr> =
            terms.into_iter().map(|t| negate_term(&t)).collect();
          build_sum(negated)
        }
        _ => Expr::UnaryOp {
          op: *op,
          operand: Box::new(operand_exp),
        },
      }
    }

    Expr::FunctionCall { name, args } => match name.as_str() {
      "Plus" => {
        let expanded_args: Vec<Expr> = args.iter().map(expand_expr).collect();
        let mut all_terms = Vec::new();
        for a in &expanded_args {
          all_terms.extend(collect_additive_terms(a));
        }
        build_sum(all_terms)
      }
      "Times" => {
        let expanded_args: Vec<Expr> = args.iter().map(expand_expr).collect();
        if expanded_args.is_empty() {
          return Expr::Integer(1);
        }
        let mut result = expanded_args[0].clone();
        for a in &expanded_args[1..] {
          result = distribute_product(&result, a);
        }
        result
      }
      "Power" if args.len() == 2 => {
        let base = expand_expr(&args[0]);
        let exp = expand_expr(&args[1]);
        if let Expr::Integer(n) = &exp
          && *n >= 2
          && is_sum(&base)
        {
          return expand_power(&base, *n);
        }
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![base, exp],
        }
      }
      _ => expr.clone(),
    },

    _ => expr.clone(),
  }
}

/// Check if an expression is a sum (Plus).
pub fn is_sum(expr: &Expr) -> bool {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus | BinaryOperator::Minus,
      ..
    } => true,
    Expr::FunctionCall { name, .. } if name == "Plus" => true,
    _ => false,
  }
}

/// Distribute the product of two expanded expressions.
/// If either is a sum, produce all cross-products.
pub fn distribute_product(left: &Expr, right: &Expr) -> Expr {
  let left_terms = collect_additive_terms(left);
  let right_terms = collect_additive_terms(right);

  if left_terms.len() == 1 && right_terms.len() == 1 {
    // Neither is a sum — just multiply
    return multiply_terms(&left_terms[0], &right_terms[0]);
  }

  let mut result_terms = Vec::new();
  for l in &left_terms {
    for r in &right_terms {
      result_terms.push(multiply_terms(l, r));
    }
  }
  build_sum(result_terms)
}

/// Multiply two non-sum terms (individual monomials).
pub fn multiply_terms(a: &Expr, b: &Expr) -> Expr {
  // Handle negation
  if let Expr::UnaryOp {
    op: UnaryOperator::Minus,
    operand,
  } = a
  {
    return negate_term(&multiply_terms(operand, b));
  }
  if let Expr::UnaryOp {
    op: UnaryOperator::Minus,
    operand,
  } = b
  {
    return negate_term(&multiply_terms(a, operand));
  }

  match (a, b) {
    (Expr::Integer(1), _) => b.clone(),
    (_, Expr::Integer(1)) => a.clone(),
    (Expr::Integer(0), _) | (_, Expr::Integer(0)) => Expr::Integer(0),
    (Expr::Integer(x), Expr::Integer(y)) => Expr::Integer(x * y),
    (Expr::Real(x), Expr::Real(y)) => Expr::Real(x * y),
    (Expr::Integer(x), Expr::Real(y)) | (Expr::Real(y), Expr::Integer(x)) => {
      Expr::Real(*x as f64 * y)
    }
    _ => {
      // Combine like bases: x * x → x^2, x^a * x^b → x^(a+b)
      let mut a_factors = collect_multiplicative_factors(a);
      let b_factors = collect_multiplicative_factors(b);
      a_factors.extend(b_factors);
      combine_product_factors(a_factors)
    }
  }
}

/// Combine multiplicative factors, merging like bases into powers.
/// [x, x, y] → x^2 * y
pub fn combine_product_factors(factors: Vec<Expr>) -> Expr {
  // Group factors by base, sum exponents
  let mut base_exps: Vec<(String, Expr, Expr)> = Vec::new(); // (sort_key, base, exponent)
  let mut numeric_coeff = Expr::Integer(1);

  for f in &factors {
    match f {
      Expr::Integer(_) | Expr::Real(_) => {
        numeric_coeff = multiply_exprs(&numeric_coeff, f);
      }
      _ => {
        let (base, exp) = extract_base_and_exp(f);
        let key = expr_to_string(&base);
        if let Some(entry) = base_exps.iter_mut().find(|(k, _, _)| *k == key) {
          entry.2 = add_exprs(&entry.2, &exp);
        } else {
          base_exps.push((key, base, exp));
        }
      }
    }
  }

  // Build result
  let mut result_factors: Vec<Expr> = Vec::new();
  if !matches!(&numeric_coeff, Expr::Integer(1)) {
    result_factors.push(numeric_coeff);
  }

  for (_, base, exp) in base_exps {
    let exp = simplify(exp);
    if matches!(&exp, Expr::Integer(0)) {
      continue; // x^0 = 1, skip
    } else if matches!(&exp, Expr::Integer(1)) {
      result_factors.push(base);
    } else {
      // Try to evaluate the power (e.g. I^2 → -1)
      if let Ok(simplified) =
        crate::functions::math_ast::power_ast(&[base.clone(), exp.clone()])
      {
        result_factors.push(simplified);
      } else {
        result_factors.push(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(base),
          right: Box::new(exp),
        });
      }
    }
  }

  if result_factors.is_empty() {
    Expr::Integer(1)
  } else {
    build_product(result_factors)
  }
}

/// Extract base and exponent from a factor.
pub fn extract_base_and_exp(expr: &Expr) -> (Expr, Expr) {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => (*left.clone(), *right.clone()),
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      (args[0].clone(), args[1].clone())
    }
    _ => (expr.clone(), Expr::Integer(1)),
  }
}

/// Negate a term.
pub fn negate_term(t: &Expr) -> Expr {
  match t {
    Expr::Integer(n) => Expr::Integer(-n),
    Expr::Real(f) => Expr::Real(-f),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => *operand.clone(),
    _ => Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(t.clone()),
    },
  }
}

/// Expand (sum)^n by repeated distribution.
pub fn expand_power(base: &Expr, n: i128) -> Expr {
  if n == 0 {
    return Expr::Integer(1);
  }
  if n == 1 {
    return base.clone();
  }
  // Repeated multiplication
  let mut result = base.clone();
  for _ in 1..n {
    result = distribute_product(&result, base);
    // Combine like terms to keep expression manageable
    let terms = collect_additive_terms(&result);
    result = combine_and_build(terms);
  }
  result
}

/// Build a sum (BinaryOp::Plus chain) from terms.
pub fn build_sum(terms: Vec<Expr>) -> Expr {
  if terms.is_empty() {
    return Expr::Integer(0);
  }
  let mut iter = terms.into_iter();
  let mut result = iter.next().unwrap();
  for t in iter {
    // Handle negative terms: a + (-b) stays as BinaryOp::Plus with UnaryOp::Minus
    result = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(result),
      right: Box::new(t),
    };
  }
  result
}

/// Combine like terms and sort, then build the final expression.
pub fn combine_and_build(terms: Vec<Expr>) -> Expr {
  // Represent each term as (key, coefficient) where key identifies the "variable part"
  let mut term_map: Vec<(String, Vec<Expr>, Expr)> = Vec::new(); // (sort_key, var_factors, coeff)

  for term in &terms {
    let (coeff, var_key, var_factors) = decompose_term(term);
    // Find existing entry
    if let Some(entry) = term_map.iter_mut().find(|(k, _, _)| *k == var_key) {
      entry.2 = add_exprs(&entry.2, &coeff);
    } else {
      term_map.push((var_key, var_factors, coeff));
    }
  }

  // Sort terms using Wolfram's canonical ordering:
  // Reverse-variable lexicographic ascending — sort by last variable ascending,
  // then next-to-last ascending, etc. Constants come first naturally (all exponents 0).
  term_map.sort_by(|(ka, va, _), (kb, vb, _)| {
    // Constants first
    match (ka.is_empty(), kb.is_empty()) {
      (true, true) => return std::cmp::Ordering::Equal,
      (true, false) => return std::cmp::Ordering::Less,
      (false, true) => return std::cmp::Ordering::Greater,
      _ => {}
    }
    let ea = extract_exponent_map(va);
    let eb = extract_exponent_map(vb);
    // Collect all variable names, sort alphabetically
    let mut all_vars: Vec<&String> = ea.keys().chain(eb.keys()).collect();
    all_vars.sort();
    all_vars.dedup();
    // Compare from LAST variable ascending, then next-to-last ascending, etc.
    for var in all_vars.iter().rev() {
      let pa = ea.get(*var).copied().unwrap_or(0);
      let pb = eb.get(*var).copied().unwrap_or(0);
      if pa != pb {
        return pa.cmp(&pb); // ascending
      }
    }
    std::cmp::Ordering::Equal
  });

  // Build result terms
  let mut result_terms: Vec<Expr> = Vec::new();
  for (_, var_factors, coeff) in term_map {
    let coeff = simplify(coeff);
    if matches!(&coeff, Expr::Integer(0)) {
      continue; // skip zero terms
    }
    if var_factors.is_empty() {
      // Constant term
      result_terms.push(coeff);
    } else if matches!(&coeff, Expr::Integer(1)) {
      // Coefficient is 1, just use the variable part
      let var_expr = build_product(var_factors);
      result_terms.push(var_expr);
    } else if matches!(&coeff, Expr::Integer(-1)) {
      let var_expr = build_product(var_factors);
      result_terms.push(negate_term(&var_expr));
    } else {
      let var_expr = build_product(var_factors);
      result_terms.push(multiply_exprs(&coeff, &var_expr));
    }
  }

  if result_terms.is_empty() {
    Expr::Integer(0)
  } else {
    build_sum(result_terms)
  }
}

/// Decompose a term into (numeric_coefficient, sort_key, variable_factors).
/// E.g. 3*x^2*y → (3, "x^2*y", [x^2, y])
///      -x → (-1, "x^1", [x])
///      5 → (5, "", [])
pub(super) fn decompose_term(term: &Expr) -> (Expr, String, Vec<Expr>) {
  match term {
    Expr::Integer(_) | Expr::Real(_) => (term.clone(), String::new(), vec![]),
    Expr::Identifier(_) => {
      (Expr::Integer(1), expr_to_string(term), vec![term.clone()])
    }
    Expr::Constant(_) => {
      (Expr::Integer(1), expr_to_string(term), vec![term.clone()])
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let (c, k, v) = decompose_term(operand);
      (negate_term(&c), k, v)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      ..
    } => {
      let factors = collect_multiplicative_factors(term);
      let mut numeric_coeff = Expr::Integer(1);
      let mut var_factors: Vec<Expr> = Vec::new();

      for f in &factors {
        match f {
          Expr::Integer(_) | Expr::Real(_) => {
            numeric_coeff = multiply_exprs(&numeric_coeff, f);
          }
          Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand,
          } => {
            numeric_coeff = negate_term(&numeric_coeff);
            match operand.as_ref() {
              Expr::Integer(_) | Expr::Real(_) => {
                numeric_coeff = multiply_exprs(&numeric_coeff, operand);
              }
              _ => var_factors.push(*operand.clone()),
            }
          }
          _ => var_factors.push(f.clone()),
        }
      }

      // Sort variable factors for canonical key
      var_factors.sort_by_key(expr_to_string);
      let key = var_factors
        .iter()
        .map(expr_to_string)
        .collect::<Vec<_>>()
        .join("*");
      (numeric_coeff, key, var_factors)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      ..
    } => (Expr::Integer(1), expr_to_string(term), vec![term.clone()]),
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut numeric_coeff = Expr::Integer(1);
      let mut var_factors: Vec<Expr> = Vec::new();

      for f in args {
        match f {
          Expr::Integer(_) | Expr::Real(_) => {
            numeric_coeff = multiply_exprs(&numeric_coeff, f);
          }
          _ => var_factors.push(f.clone()),
        }
      }

      var_factors.sort_by_key(expr_to_string);
      let key = var_factors
        .iter()
        .map(expr_to_string)
        .collect::<Vec<_>>()
        .join("*");
      (numeric_coeff, key, var_factors)
    }
    _ => (Expr::Integer(1), expr_to_string(term), vec![term.clone()]),
  }
}

/// Collect multiplicative factors from nested Times.
pub fn collect_multiplicative_factors(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let mut factors = collect_multiplicative_factors(left);
      factors.extend(collect_multiplicative_factors(right));
      factors
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut factors = Vec::new();
      for a in args {
        factors.extend(collect_multiplicative_factors(a));
      }
      factors
    }
    _ => vec![expr.clone()],
  }
}

/// Build a product from factors.
pub fn build_product(factors: Vec<Expr>) -> Expr {
  if factors.is_empty() {
    return Expr::Integer(1);
  }
  let mut iter = factors.into_iter();
  let mut result = iter.next().unwrap();
  for f in iter {
    result = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(result),
      right: Box::new(f),
    };
  }
  result
}

// ─── ExpandAll ──────────────────────────────────────────────────────

/// ExpandAll[expr] - Recursively expands all subexpressions
pub fn expand_all_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ExpandAll expects exactly 1 argument".into(),
    ));
  }
  Ok(expand_all_recursive(&args[0]))
}

/// Recursively expand all subexpressions, then expand at the top level
pub fn expand_all_recursive(expr: &Expr) -> Expr {
  match expr {
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::String(_)
    | Expr::Constant(_)
    | Expr::Identifier(_)
    | Expr::Slot(_) => expr.clone(),

    Expr::BinaryOp { op, left, right } => {
      let left_exp = expand_all_recursive(left);
      let right_exp = expand_all_recursive(right);
      // After recursively expanding sub-expressions, expand at this level
      expand_and_combine(&Expr::BinaryOp {
        op: *op,
        left: Box::new(left_exp),
        right: Box::new(right_exp),
      })
    }

    Expr::UnaryOp { op, operand } => {
      let operand_exp = expand_all_recursive(operand);
      expand_and_combine(&Expr::UnaryOp {
        op: *op,
        operand: Box::new(operand_exp),
      })
    }

    Expr::FunctionCall { name, args } => {
      let expanded_args: Vec<Expr> =
        args.iter().map(expand_all_recursive).collect();
      // After expanding sub-expressions, expand at this level for Plus/Times/Power
      match name.as_str() {
        "Plus" | "Times" | "Power" => expand_and_combine(&Expr::FunctionCall {
          name: name.clone(),
          args: expanded_args,
        }),
        _ => Expr::FunctionCall {
          name: name.clone(),
          args: expanded_args,
        },
      }
    }

    _ => expr.clone(),
  }
}
